import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_PA as vit
import math

from vilt.modules.vision_transformer_PA import Adapter_Layer as ParallelAdapter
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

def tensor_in_list(tensor_list, new_tensor):
    for tensor in tensor_list:
        if torch.equal(tensor, new_tensor):
            return True
    return False

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, 
                 config, 
                 get_val_loader,
                 trainable=["classifier", "pooler", "token_type_embeddings", "rank_output", "adapter"]):
        super().__init__()
        self.save_hyperparameters(ignore='get_val_loader')

        self.get_val_loader = get_val_loader
        self.das_val_size = config["per_gpu_batchsize"]

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.adapter_list = nn.ModuleList([
            nn.Sequential(
                ParallelAdapter(bottleneck = 96 * 2),
            ) for _ in range(12)])
        self.skip_flag = torch.ones(12) * -1.0

        self.register_buffer('skip_num', torch.ones(1) * config["skip_num"])
        self.register_buffer('das_gate', torch.zeros(12))

        self.warmup_epoch = 1
        self.das_epoch = config['das_epoch']
        self.das_step = config['das_step']
        self.das_turn = config['das_turn']
        self.das_count = 0

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        self.trainable = trainable
        for n, p in self.named_parameters():
            # print(n)
            if not any(t in n for t in self.trainable):
                p.requires_grad = False
            else:
                print(n)

        orig_param_size = sum(p.numel() for p in self.parameters())
        trainable_size =  sum(p.numel() for p in self.parameters() if p.requires_grad)
        extra_param = sum(p.numel() for n, p in self.named_parameters() if "adapter" in n)
        print('extra parameter:{}'.format(extra_param))
        print('trainable_size:{:.4f}%({}/{})'.format(trainable_size / orig_param_size * 100, trainable_size, orig_param_size))

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

            print()
            print(select)
            print('Fusion Image', self.das_gate[:6])
            print('Fusion Text ', self.das_gate[6: 12])
            print('Encoder Text', self.das_gate[12:])

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].cuda()
        text_labels = batch[f"text_labels{do_mlm}"].cuda()
        text_masks = batch[f"text_masks"].cuda()
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0].cuda()
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            if self.skip_flag[i] > 0:
                x = self.adapter_list[i](x)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret
    
    def calculate_loss(self, ret, batch):
        ret = dict()

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def clean_flag(self):
        for i in range(12):
            self.skip_flag[i] = -1.0

    def apply_flag(self, select):
        for i in select:
            i = i.item()
            self.skip_flag[i] = 1.0

    def get_prob(self):
        # prob = torch.softmax(self.das_gate * 4 ** self.current_epoch, dim=-1)
        prob = torch.sigmoid(self.das_gate)

        return prob

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if self.training:
            if self.current_epoch < self.das_epoch:
                self.clean_flag()

                prob = self.get_prob()
                select = torch.multinomial(prob, round(self.skip_num.item()))
                self.apply_flag(select)
                self.das_count += 1

                ret = self.calculate_loss(ret, batch)

                if self.das_count >= self.das_turn and self.current_epoch >= self.warmup_epoch:
                    val_loader = self.get_val_loader(self.das_val_size)
                    val_batch = next(iter(val_loader))

                    rets = []
                    selects = []
                    self.das_count = 0

                    prob = self.get_prob()
                    for k in range(self.das_step):
                        self.clean_flag()

                        select = torch.sort(torch.multinomial(prob, round(self.skip_num.item())))[0]
                        while tensor_in_list(selects, select):
                            select = torch.sort(torch.multinomial(prob, round(self.skip_num.item())))[0]
                        selects.append(select)

                        self.apply_flag(select)

                        val_ret = dict()
                        with torch.no_grad():
                            val_ret = self.calculate_loss(val_ret, val_batch)

                        rets.append(val_ret)

                    rewards = []
                    for i in range(self.das_step):
                        rewards.append(math.exp(-sum([v.item() for k, v in rets[i].items() if "loss" in k])))
                    rewardb = sum(rewards) / self.das_step

                    lr = 1.0 # 1.0 for VQA
                    for k in range(self.das_step):
                        for i in selects[k]:
                            i = i.item()
                            self.das_gate[i] += lr * (rewards[k] - rewardb) * prob[i] * (1 - prob[i])                
            else:
                ret = self.calculate_loss(ret, batch)
        else:
            ret = self.calculate_loss(ret, batch) 

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

        self.clean_flag()

        select = torch.sort(self.das_gate)[1]
        select = select[-round(self.skip_num.item()):]
        print()
        print(select)
        self.apply_flag(select)

    # def validation_step(self, batch, batch_idx):
    #     vilt_utils.set_task(self)
    #     output = self(batch)

    # def validation_epoch_end(self, outs):
    #     vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)