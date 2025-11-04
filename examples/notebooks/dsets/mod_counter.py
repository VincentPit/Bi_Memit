import json
import typing
from pathlib import Path
import requests

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

class ModCFDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/scratch/jl13122/generated_counterfact_dataset",
        tok: AutoTokenizer = None,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        A PyTorch Dataset class for loading the ZSRE modified dataset.
        
        Args:
            data_dir (str): The directory where the dataset is located.
            tok (AutoTokenizer): Tokenizer instance for processing text.
            size (Optional[int]): Maximum number of data items to load. Defaults to None (loads all data).
        """
        data_dir = Path(data_dir)
        data_file = data_dir / "train.json"
        locality_file = data_dir / "locality_questions.json"
        generality_file = data_dir / "generality_questions.json"
        
        # Check if required files exist
        if not data_file.exists():
            raise FileNotFoundError(f"{data_file} does not exist.")
        if not locality_file.exists():
            raise FileNotFoundError(f"{locality_file} does not exist.")
        if not generality_file.exists():
            raise FileNotFoundError(f"{generality_file} does not exist.")

        # Load data from files
        with open(data_file, "r") as f:
            raw = json.load(f)

        with open(locality_file, "r") as f:
            locality_data = json.load(f)

        with open(generality_file, "r") as f:
            generality_data = json.load(f)

        # Create dictionaries for quick lookup of locality and generality questions
        locality_questions = self._group_by_case_id(locality_data)
        generality_questions = self._group_by_case_id(generality_data)

        # Limit size if specified
        if size is not None:
            raw = raw[:size]

        print(f"Loaded dataset with {len(raw)} elements from {data_file}")
        
        # Process and structure the data
        data = []
        for i, record in enumerate(raw):
            # Tokenize the 'object_true' (assumed here as 'loc_ans')
            ans_toks = tok(" " + record["object_true"])["input_ids"]

            # Get locality and generality questions for the current case_id
            case_id = record["case_id"]
            local_prompts = locality_questions.get(case_id, [])
            general_prompts = generality_questions.get(case_id, [])

            # Clean and structure the data
            
            
            """if record["subject"] != "People's Republic of Poland":
                continue"""
            
            #print("subject:", self._clean(record["subject"]))
            
            subject_id = self.get_wikidata_id(record["subject"])
            subject = self._clean(record["subject"])
            data.append(
                {
                    "case_id": case_id,
                    "requested_rewrite": {
                        "prompt": self._clean_and_replace(
                            record["prompt_for_object_false"][0], 
                            subject, 
                            "{}",
                            second_only=True
                        ),
                        "subject": record["subject"],
                        "target_new": {"str": self._clean(record["object_false"]), "id": record["object_false_id"]},
                        "target_true": {"str": self._clean(record["object_true"]), "id": record["object_false_id"]},
                        "relation_id":  record["relation_id"],
                    },
                    "requested_reverse_rewrite": {
                        "prompt": self._clean_and_replace(
                            record["prompt_for_subject"][0],
                            self._clean(record["object_false"]),
                            "{}",
                            second_only=True
                        ),
                        "subject": record["object_false"],
                        "target_new": {"str": subject, "id": subject_id},
                        "target_true": {"str": subject, "id": subject_id},
                        "relation_id":  record["relation_id"],
                    },
                    
                    
                    "paraphrase_prompts": [self._clean(p) for p in (record.get("prompt_for_object_false", [])[:2])],
                    "reverse_paraphrase_prompts": [self._clean(p) for p in record.get("prompt_for_subject", [])],
                    "local_prompts": [self._clean(p) for p in local_prompts],
                    
                    "relation_prompt": [self._clean(p) for p in record.get("prompt_for_relation", [])],
                    
                    
                    
                    "general_prompts": [self._clean(p) for p in general_prompts],
                    "generation_prompts": [self._clean(p) for p in record.get("prompt_for_object_false", [])[2:]],
                    "neighborhood_prompts": [self._clean(p) for p in local_prompts],
                    "attribute_prompts": [self._clean(p) for p in local_prompts],

                }
            )
            # Stop if size limit is reached
            if size and len(data) >= size:
                break

        random.shuffle(data)

        self._data = data
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @staticmethod
    def _group_by_case_id(data):
        """
        Groups a list of dictionaries by their 'case_id'.

        Args:
            data (list): List of dictionaries with 'case_id' and 'question'.

        Returns:
            dict: A dictionary where keys are case_ids and values are lists of questions.
        """
        grouped = {}
        for item in data:
            case_id = item["case_id"]
            if case_id not in grouped:
                grouped[case_id] = []
            grouped[case_id].append(item["question"])
        return grouped
    
    import requests
    
    
    def get_wikidata_id(self, subject):
        """
        Fetches the Wikidata ID for a given subject using the Wikidata SPARQL endpoint.
        
        Args:
            subject (str): The name of the subject to query.
            
        Returns:
            str: The Wikidata ID of the subject if found, else None.
        """
        query = """
        SELECT ?item WHERE {
        ?item ?label "%s"@en.
        } LIMIT 1
        """ % subject

        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/json"}
        params = {"query": query}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {}).get("bindings", [])
            if results:
                return results[0]["item"]["value"].split("/")[-1]  # Extract the ID from the URL
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        return None


    @staticmethod
    def _clean(text):
        """
        Cleans the text by removing unwanted characters and trimming whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        allowed_chars = set(",.+-|\\/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ?")
        cleaned_text = ''.join(c if c in allowed_chars else '' for c in text)
        return cleaned_text.strip()

    @staticmethod
    def _clean_and_replace(text, old, new, second_only=False):
        """
        Cleans the text and replaces the target word with a new one.

        Args:
            text (str): The text to process.
            old (str): The word to replace.
            new (str): The word to replace with.
            second_only (bool): If True, replaces only the second occurrence.

        Returns:
            str: The processed text.
        """
        
        
        text = ModCFDataset._clean(text)
        assert old in text, f"[{text}] does not have item {old} in it. Try to skip maybe"
        if second_only:
            parts = text.split(old, 2)
            if len(parts) == 3:
                return parts[0] + old + parts[1] + new + parts[2]
        return text.replace(old, new)
