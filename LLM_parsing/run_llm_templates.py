import os
import sys
import pandas as pd
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_parsing.llm import generate_json_batch
from prompts.dimensions import dim_prompts
from prompts.date import date_prompts
from prompts.artist import artist_prompts

def clean_json_response(parsed: dict) -> dict:
    if not isinstance(parsed, dict) or not parsed:
        return None
        
    return parsed
        
def main():
    parser = argparse.ArgumentParser(description="Run LLM batches for unique attribute templates.")
    parser.add_argument("--task", required=True, choices=['dimensions', 'date', 'artist'], help="Attribute type to run.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to top N templates.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum frequency of template to process.")
    parser.add_argument("--max-freq", type=int, default=None, help="Maximum frequency of template to process.")
    parser.add_argument("--model", type=str, default="g25flash", help="Model to use (g25flash, g25lite).")
    parser.add_argument("--no-cache", action="store_true", help="Bypass LLM cache and force fresh generation.")
    args = parser.parse_args()

    templates_csv = f'LLM_parsing/llm_unparsed_{args.task}_templates.csv'
    if not os.path.exists(templates_csv):
        print(f"Error: {templates_csv} not found. Run generate_llm_templates.py first.")
        return
        
    df = pd.read_csv(templates_csv)
    
    mask = (df['frequency'] >= args.min_freq)
    if args.max_freq:
        mask &= (df['frequency'] <= args.max_freq)
        
    filtered_df = df[mask].copy()
    
    if args.limit:
        filtered_df = filtered_df.head(args.limit)
        
    print(f"Loaded {len(df)} templates for {args.task}. Filtering for freq {args.min_freq}-{args.max_freq or 'inf'} yields {len(filtered_df)} items.")
    
    if args.task == 'dimensions':
        prompt_template = dim_prompts.get("general", list(dim_prompts.values())[0])
        input_key = "dimensions"
    elif args.task == 'date':
        prompt_template = date_prompts.get("artist", list(date_prompts.values())[0])
        input_key = "date"
    elif args.task == 'artist':
        prompt_template = artist_prompts.get("artist", list(artist_prompts.values())[0])
        input_key = "artist_information"

    message_batch = []
    template_strings = []
    
    for _, row in filtered_df.iterrows():
        example_str = row['example_raw_string']
        template_str = row['template']
        
        format_kwargs = {input_key: example_str}
        if args.task == 'artist':
            format_kwargs['artist_titles'] = ""
            
        try:
            prompt = prompt_template.format(**format_kwargs)
        except KeyError as e:
            print(f"Warning: Prompt formatting failed for {template_str}: {e}")
            continue

        messages = [
            {"role": "user", "content": prompt}
        ]
        message_batch.append(messages)
        template_strings.append(template_str)
        
    if not message_batch:
        print("No templates to process.")
        return

    output_file = f'LLM_parsing/llm_{args.task}_mapping.json'
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            output_mapping = json.load(f)
    else:
        output_mapping = {}

    print(f"Sending {len(message_batch)} {args.task} templates to {args.model} (no_cache={args.no_cache})...")
    
    sub_batch_size = 100
    total_input_tokens = 0
    total_output_tokens = 0
    success_count = 0
    
    for i in range(0, len(message_batch), sub_batch_size):
        sub_messages = message_batch[i:i + sub_batch_size]
        sub_templates = template_strings[i:i + sub_batch_size]
        
        current_batch_num = (i // sub_batch_size) + 1
        total_batches = (len(message_batch) + sub_batch_size - 1) // sub_batch_size
        print(f"\n--- Sub-batch {current_batch_num}/{total_batches} ({len(sub_messages)} templates) ---")
        
        results = generate_json_batch(
            message_list=sub_messages,
            model=args.model,
            fallback_model="g25lite",
            max_workers=10,
            chunk_size=20,
            rpm_limit=1000,
            ignore_cache=args.no_cache
        )
        
        sub_success = 0
        for template_str, result_tuple in zip(sub_templates, results):
            if not result_tuple:
                continue
                
            parsed_dict, usage = result_tuple
            
            if usage:
                if hasattr(usage, "prompt_tokens"):
                    total_input_tokens += (usage.prompt_tokens or 0)
                    total_output_tokens += (usage.completion_tokens or 0)
                elif isinstance(usage, dict):
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)
            
            parsed_json = clean_json_response(parsed_dict)
            if parsed_json:
                output_mapping[template_str] = parsed_json
                sub_success += 1
                success_count += 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_mapping, f, indent=2)
        print(f"  Processed sub-batch: {sub_success} successes. Mapping saved to {output_file}")
                
    print(f"\n--- BATCH EXECUTION SUMMARY ({args.task}) ---")
    print(f"Total Inputs Sent: {len(template_strings)}")
    print(f"Successfully Parsed JSONs: {success_count} ({(success_count/len(template_strings))*100:.2f}%)")
    print(f"Approx Token Usage: {total_input_tokens} Input, {total_output_tokens} Output")
    print(f"Final Template -> JSON mappings exported to {output_file}")

if __name__ == "__main__":
    main()
