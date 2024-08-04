from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel, XLMRobertaConfig
import torch

class XLMRobertaForRegression(XLMRobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    auto_map = {"AutoModel": "model.XLMRobertaForRegression"}

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.roberta = XLMRobertaModel(config)
        self.regression_head = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()
        self.original_params = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, num_unfrozen_layers=0, **kwargs):
        # Use the parent class method to load the pretrained weights
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # After loading pretrained weights, set requires_grad and original_params
        model.set_requires_grad(num_unfrozen_layers)
        model.set_original_params()
        
        return model

    def set_original_params(self):
        """Set original_params after loading pretrained weights."""
        self.original_params = {
            name: param.data.clone()
            for name, param in self.roberta.named_parameters()
            if param.requires_grad
        }

    def set_requires_grad(self, num_unfrozen_layers):
        """Set requires_grad after loading pretrained weights."""
        # Freeze all layers
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Unfreeze the top num_unfrozen_layers
        if num_unfrozen_layers > 0:
            for param in self.roberta.encoder.layer[-num_unfrozen_layers:].parameters():
                param.requires_grad = True

        # Always unfreeze the regression head
        for param in self.regression_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token output
        logit = self.regression_head(cls_output)
        out = torch.nn.functional.sigmoid(logit)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(out.squeeze(), labels.squeeze())

        return (loss, out) if loss is not None else out

    def get_param_regularization_loss(self, lambda_reg):
        reg_loss = torch.tensor([0.0]).to(self.device)
        for name, param in self.roberta.named_parameters():
            if param.requires_grad and name in self.original_params:
                reg_loss += torch.sum((param - self.original_params[name]) ** 2)
        return lambda_reg * reg_loss / (sum(p.numel() for p in self.roberta.parameters() if p.requires_grad) + 1e-8)

def create_model(config, num_unfrozen_layers=3):
    return XLMRobertaForRegression(config, num_unfrozen_layers=num_unfrozen_layers)