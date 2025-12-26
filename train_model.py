import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModel, 
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    BertPreTrainedModel,
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from transformers.modeling_outputs import MaskedLMOutput
import os

OUTPUT_DIR = "./scibert-condenser-cord19-final"

# ARCHITECTURE FOR CONDENSER 
class CondenserForPretraining(BertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.bias"]  
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)

        # Config Condenser: Skip from layer 6 to layer 12
        self.skip_from = 6

        # Prediction head for MLM task
        self.cls = AutoModelForMaskedLM.from_config(config).cls
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Run BERT backbone
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # Take Hidden state from layer skip (Layer 6)
        skip_layer_h = outputs.hidden_states[self.skip_from + 1]
        
        # Take CLS from layer skip
        cls_h = skip_layer_h[:, :1, :] 
        
        # (Layer 12)
        rest_h = outputs.last_hidden_state[:, 1:, :] 
        
        # CLS (word skip) + Content (last layer)
        condenser_input = torch.cat([cls_h, rest_h], dim=1)
        
        # Calculate MLM 
        prediction_scores = self.cls(condenser_input)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# CONFIG
model_name = "allenai/scibert_scivocab_uncased" 
dataset_name = "manhngvu/cord19_chunked_300_words"

print(f"Init Condenser from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Load model 
model = CondenserForPretraining.from_pretrained(model_name, config=config)
model.train() 

# DATASET 
print(f"Loading dataset {dataset_name}...")
ds = load_dataset(dataset_name, split="train")

# use this for testing purpose only
# ds = ds.select(range(1000)) 

def tokenize_function(examples):
    return tokenizer(
        examples["chunk_text"], 
        truncation=True, 
        padding="max_length", 
        max_length=256 
    )

print("Tokenizing data ...")
tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names, num_proc=4)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

print("Start Training")

training_args = TrainingArguments(
    output_dir= OUTPUT_DIR, 
    overwrite_output_dir=True,
    num_train_epochs=3,             
    per_device_train_batch_size=16,  
    save_steps=500,
    save_total_limit=5,
    learning_rate=1e-4,              
    weight_decay=0.01,
    fp16=False,                      
    use_mps_device=True,            
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",
    save_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# HANDLE CHECKPOINTS
last_checkpoint = None 
if os.path.isdir(OUTPUT_DIR):
    from transformers.trainer_utils import get_last_checkpoint
    try:
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f"Found checkpoint: {last_checkpoint}")
            print(f"Continue training from checkpoint...")
        else:
            print(f"Not have check point")
    except Exception as e:
        print(f"Error checkpoint: {e}")
        last_checkpoint = None

trainer.train(resume_from_checkpoint=last_checkpoint)

# Save model
print("Saving model...")
trainer.save_model("./scibert-condenser-cord19-final")
tokenizer.save_pretrained("./scibert-condenser-cord19-final")

# Save Backbone for after steps (Fine-tuning)
model.bert.save_pretrained("./condenser_backbone", safe_serialization=False)
tokenizer.save_pretrained("./condenser_backbone")

print("Doneeeeee")
