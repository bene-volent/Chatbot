from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch
import re
from typing import List, Dict, Any, Optional, Union
import os

class BaseModelHandler:
    """Base class for all model handlers"""
    
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.max_new_tokens = 512
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the model and tokenizer - to be implemented by subclasses"""
        raise NotImplementedError
    
    def warm_up(self):
        """Perform a warm-up pass to initialize the model"""
        print(f"Performing warm-up pass for {self.model_name}...")
        dummy_input = self.tokenizer("Warm-up pass", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(**dummy_input, max_length=10)
        print("Model ready!")
    
    def format_conversation(self, history: str, new_message: str) -> str:
        """Format the conversation history and new message for the model"""
        raise NotImplementedError
    
    def get_stopping_criteria(self) -> StoppingCriteriaList:
        """Get the stopping criteria for text generation"""
        return StoppingCriteriaList([
            UserMessageStoppingCriteria(self.tokenizer)
        ])
    
    def get_generation_kwargs(self, inputs: Dict[str, torch.Tensor], streamer: TextIteratorStreamer) -> Dict[str, Any]:
        """Get the keyword arguments for text generation"""
        input_length = len(inputs["input_ids"][0])
        dynamic_max_length = min(len(inputs["input_ids"][0]) + self.max_new_tokens, self.tokenizer.model_max_length)
        
        return {
            "input_ids": inputs["input_ids"],
            "max_length": dynamic_max_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "stopping_criteria": self.get_stopping_criteria(),
            "streamer": streamer,
            "attention_mask": inputs.get("attention_mask", None)
        }
    
    def clean_response(self, text: str) -> str:
        """Clean up response text from unwanted artifacts"""
        # Remove any <think> or </think> tags
        text = re.sub(r'</?think>', '', text)
        
        # Remove any System: prefixes that got generated
        text = re.sub(r'^System:', '', text)
        
        # Remove any lines that start with System:, User: or Assistant: that aren't part of the conversation
        text = re.sub(r'\n(System|User|Assistant):[^\n]*', '', text)
        
        # Check for incomplete sentences
        last_sentence = text.split('.')[-1]
        if len(last_sentence) > 30 and not any(text.endswith(c) for c in ['.', '!', '?', ':', ';']):
            # Likely an incomplete sentence - add ellipsis
            text += "..."
            
        # Remove end markers
        text = re.sub(r'\[END\]', '', text)
        text = re.sub(r'<STOP>', '', text)
        text = re.sub(r'</s>', '', text)
        
        # Clean up model-specific artifacts
        text = self._model_specific_cleaning(text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _model_specific_cleaning(self, text: str) -> str:
        """Model-specific cleaning logic to be implemented by subclasses"""
        return text

class DeepSeekHandler(BaseModelHandler):
    """Handler for DeepSeek models"""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device: str = None):
        super().__init__(device)
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.warm_up()
    
    def format_conversation(self, history: str, new_message: str) -> str:
        system_message = ("You are a helpful, concise assistant. Provide clear and accurate answers. "
                         "Don't include any internal thoughts or repeat the user's message. "
                         "Always complete your sentences and thoughts fully. "
                         "After you've completed your full response, add '[END]' on a new line.")
        
        if not history:
            formatted = f"System: {system_message}\nUser: {new_message}\nAssistant:"
        else:
            formatted = f"{history}\nUser: {new_message}\nAssistant:"
        return formatted

class Phi2Handler(BaseModelHandler):
    """Handler for Microsoft's Phi-2 model"""
    
    def __init__(self, model_name: str = "microsoft/phi-2", device: str = None):
        super().__init__(device)
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.warm_up()
    
    def format_conversation(self, history: str, new_message: str) -> str:
        system_message = ("You are a helpful, concise education assistant specialized in NCERT textbooks. "
                         "Provide clear and accurate explanations suitable for students. "
                         "After you've completed your full response, add '[END]' on a new line.")
        
        if not history:
            formatted = f"<|user|>\n{system_message}\n\n{new_message}\n<|assistant|>\n"
        else:
            # Format continuing conversations based on Phi-2's preferred format
            # For Phi-2, we'll extract the last message and format it properly
            formatted = f"{history}\n<|user|>\n{new_message}\n<|assistant|>\n"
        return formatted
    
    def _model_specific_cleaning(self, text: str) -> str:
        # Clean up Phi-2 specific tags
        text = re.sub(r'<\|assistant\|>', '', text)
        text = re.sub(r'<\|user\|>', '', text)
        return text

class GemmaHandler(BaseModelHandler):
    """Handler for Google's Gemma models"""
    
    def __init__(self, model_name: str = "google/gemma-2b", device: str = None):
        super().__init__(device)
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.warm_up()
    
    def format_conversation(self, history: str, new_message: str) -> str:
        system_message = ("You are a helpful, concise education assistant specialized in NCERT textbooks. "
                         "Provide clear and accurate explanations suitable for students. "
                         "After you've completed your full response, add '[END]' on a new line.")
        
        if not history:
            # Gemma uses a specific format for instructions
            formatted = f"<start_of_turn>user\n{system_message}\n\n{new_message}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted = f"{history}\n<start_of_turn>user\n{new_message}<end_of_turn>\n<start_of_turn>model\n"
        return formatted
    
    def _model_specific_cleaning(self, text: str) -> str:
        # Clean up Gemma specific tags
        text = re.sub(r'<start_of_turn>model', '', text)
        text = re.sub(r'<end_of_turn>', '', text)
        text = re.sub(r'<start_of_turn>user', '', text)
        return text

class MistralQuantizedHandler(BaseModelHandler):
    """Handler for quantized Mistral models"""
    
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ", device: str = None):
        super().__init__(device)
        self.model_name = model_name
        try:
            # Try to import needed libraries for GPTQ models
            from optimum.gptq import GPTQQuantizer
            self.load_model()
        except ImportError:
            print("Error: To use quantized models, install optimum package: pip install optimum")
            raise
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.warm_up()
    
    def format_conversation(self, history: str, new_message: str) -> str:
        system_message = ("You are a helpful, concise education assistant specialized in NCERT textbooks. "
                         "Provide clear and accurate explanations suitable for students. "
                         "After you've completed your full response, add '[END]' on a new line.")
        
        if not history:
            # Mistral uses a specific chat format
            formatted = f"<s>[INST] {system_message}\n\n{new_message} [/INST]"
        else:
            # For continuing conversations
            formatted = f"{history}\n[INST] {new_message} [/INST]"
        return formatted
    
    def _model_specific_cleaning(self, text: str) -> str:
        # Clean up Mistral specific tags
        text = re.sub(r'\[INST\]', '', text)
        text = re.sub(r'\[/INST\]', '', text)
        return text
    
class TinyLlamaHandler(BaseModelHandler):
    """Handler for TinyLlama models"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = None):
        super().__init__(device)
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.warm_up()
        
        # TinyLlama is small, so we can use more tokens
        self.max_new_tokens = 768
    
    def format_conversation(self, history: str, new_message: str) -> str:
        system_message = ("You are a helpful, concise assistant. Provide clear and accurate answers. "
                         "Don't include any internal thoughts or repeat the user's message. "
                         "Always complete your sentences and thoughts fully. "
                         "After you've completed your full response, add '[END]' on a new line.")
        
        if not history:
            # TinyLlama uses Llama-style chat format
            formatted = f"<|system|>\n{system_message}\n<|user|>\n{new_message}\n<|assistant|>\n"
        else:
            # For continuing conversations
            formatted = f"{history}\n<|user|>\n{new_message}\n<|assistant|>\n"
        return formatted
    
    def _model_specific_cleaning(self, text: str) -> str:
        # Clean up TinyLlama specific tags
        text = re.sub(r'<\|system\|>', '', text)
        text = re.sub(r'<\|user\|>', '', text)
        text = re.sub(r'<\|assistant\|>', '', text)
        return text
    
    def get_generation_kwargs(self, inputs: Dict[str, torch.Tensor], streamer: TextIteratorStreamer) -> Dict[str, Any]:
        """Customize generation parameters for TinyLlama"""
        input_length = len(inputs["input_ids"][0])
        dynamic_max_length = min(len(inputs["input_ids"][0]) + self.max_new_tokens, self.tokenizer.model_max_length)
        
        return {
            "input_ids": inputs["input_ids"],
            "max_length": dynamic_max_length,
            "do_sample": True,
            "temperature": 0.8,  # Slightly higher temperature for TinyLlama
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "stopping_criteria": self.get_stopping_criteria(),
            "streamer": streamer,
            "attention_mask": inputs.get("attention_mask", None)
        }

# Factory class to get the right model handler
class ModelHandlerFactory:
    """Factory class to create appropriate model handlers"""
    
    @staticmethod
    def get_handler(model_name: str = None, device: str = None) -> BaseModelHandler:
        """Get the appropriate handler for the specified model"""
        
        # If no model specified, look for environment variable
        if not model_name:
            model_name = os.environ.get("LLM_MODEL", "microsoft/phi-2")
        
        # Check model name and return appropriate handler
        model_name_lower = model_name.lower()
        
        if "phi-2" in model_name_lower or "phi2" in model_name_lower:
            return Phi2Handler(model_name, device)
        elif "gemma" in model_name_lower:
            return GemmaHandler(model_name, device)
        elif "mistral" in model_name_lower and ("gptq" in model_name_lower or "quant" in model_name_lower):
            return MistralQuantizedHandler(model_name, device)
        elif "tinyllama" in model_name_lower:
            return TinyLlamaHandler(model_name, device)
        elif "deepseek" in model_name_lower:
            return DeepSeekHandler(model_name, device)
        else:
            # Default to DeepSeek handler if model type can't be determined
            print(f"Unknown model type: {model_name}. Using DeepSeek handler.")
            return DeepSeekHandler(model_name, device)

# Custom stopping criteria to prevent model from continuing as user
class UserMessageStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings=["User:", "\nUser", "System:", "\nSystem", "<STOP>", 
                                             "[END]", "</s>", "<end_of_turn>"]):
        self.tokenizer = tokenizer
        self.stop_string_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
        # Process specific end markers separately for efficiency
        self.end_marker_ids = [
            tokenizer.encode("[END]", add_special_tokens=False),
            tokenizer.encode("<STOP>", add_special_tokens=False),
            tokenizer.encode("</s>", add_special_tokens=False),
            tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        ]
        self.max_length = 1000  # Maximum reasonable response length 
        self.token_count = 0
        
    def __call__(self, input_ids, scores, **kwargs):
        self.token_count += 1
        
        # Check if the response has become too long
        if self.token_count > self.max_length:
            print("Response too long, stopping.")
            return True
        
        # Check for end markers by looking for their token IDs in the recent part of the sequence
        # This is more efficient than decoding the full text
        recent_tokens = input_ids[0, -20:].tolist()  # Only look at recent tokens
        
        # Check for end markers in a sliding window manner
        for marker_ids in self.end_marker_ids:
            for i in range(len(recent_tokens) - len(marker_ids) + 1):
                if recent_tokens[i:i+len(marker_ids)] == marker_ids:
                    print(f"Found end marker: {self.tokenizer.decode(marker_ids)}")
                    return True
        
        # Check for other stop strings
        for stop_ids in self.stop_string_ids:
            if len(stop_ids) <= input_ids.shape[1]:
                if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                    print(f"Found stop sequence: {self.tokenizer.decode(stop_ids)}")
                    return True
                    
        return False