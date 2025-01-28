import torch

from transformers import AutoModelForCausalLM

import fiftyone as fo

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from typing import Optional

class JanusVQA:
    """
    A processor class for visual-language tasks using the Janus model.
    
    This class handles the initialization and processing of visual-language tasks using
    the Janus model, a large language model designed for multi-modal interactions.
    It specifically handles visual question-answering (VQA) tasks by processing
    images and questions to generate natural language answers.
    
    Args:
        model_path (str): Path to the pre-trained Janus model. Defaults to "deepseek-ai/Janus-Pro-7B".
        device (Optional[str]): Device to run the model on ('cuda', 'cpu', 'mps'). 
            If None, automatically selects the best available device.
        question_field (str): Field name for storing questions. Defaults to "question".
        answer_field (str): Field name for storing answers. Defaults to "answer".
    """
    def __init__(
        self, 
        model_path: str = "deepseek-ai/Janus-Pro-7B",
        question_field: str = "question",
        answer_field: str = "answer"
    ):
        self.device = self._get_device()
        self.question_field = question_field
        self.answer_field = answer_field
        
        # Initialize model components
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

        # Only use bfloat16 when on CUDA
        if self.device.type == "cuda":
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16)
        self.vl_gpt = self.vl_gpt.to(self.device).eval()

    def _get_device(self) -> torch.device:
        """Determine the appropriate device to use."""
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    def create_conversation(self, question: str, image_path: str):
        """
        Creates a conversation format required by the Janus model.
        
        Structures the input in the specific format expected by Janus, including
        the image placeholder and role markers for the conversation flow.
        
        Args:
            question (str): The question to ask about the image.
            image_path (str): Path to the image file.
            
        Returns:
            list: A list of dictionaries containing the conversation structure with
                appropriate role markers for Janus model processing.
        """
        return [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    
    def process_single_image(self, question: str, image_path: str) -> str:
        """
        Process a single image-question pair using the Janus model to generate an answer.
        
        This method handles the complete pipeline of processing an image and question:
        loading the image, preparing the inputs, generating the response through
        the Janus model, and decoding the output.
        
        Args:
            question (str): The question to ask about the image.
            image_path (str): Path to the image file.
            
        Returns:
            str: The Janus model's answer to the question about the image.
        """
        with torch.no_grad():
            conversation = self.create_conversation(question, image_path)
            
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation, 
                images=pil_images, 
                force_batchify=True
            ).to(self.device)
            
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
            
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return answer

def process_dataset(
    dataset: fo.Dataset, 
    question: str,
    model_path: str = "deepseek-ai/Janus-Pro-7B",
    question_field: str = "question",
    answer_field: str = "answer"
):
    """
    Process all samples in a FiftyOne dataset.
    
    Args:
        dataset: FiftyOne dataset to process
        question: Question to ask about each image
        model_path: Path to the model
        device: Device to run the model on ('cuda', 'cpu', 'mps', etc.)
        question_field: Field name to store the question
        answer_field: Field name to store the model's answer
    """
    # Initialize processor
    processor = JanusVQA(
        model_path=model_path, 
        question_field=question_field,
        answer_field=answer_field
    )
    
    for sample in dataset.iter_samples(progress=True):
        try:
            # Store the question
            sample[question_field] = question
            
            # Process image and get answer
            answer = processor.process_single_image(question, sample.filepath)
            
            # Store the answer
            sample[answer_field] = answer
        except Exception as e:
            print(f"Error processing sample {sample.id}: {str(e)}")
        sample.save()
    
    # Save the entire dataset
    dataset.save()
    print(f"Added fields '{question_field}' and '{answer_field}' to dataset")