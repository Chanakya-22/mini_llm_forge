import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def migrate_legacy_data(input_file: str, output_file: str):
    """
    Migrates legacy <human>/<bot> string datasets into strictly formatted 
    ChatML JSON Lines files.
    """
    if not os.path.exists(input_file):
        logging.error(f"Legacy dataset not found at {input_file}")
        return

    migrated_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                raw_text = data.get("text", "")
                
                # Split the legacy flat string
                if "<human>:" not in raw_text or "<bot>:" not in raw_text:
                    logging.warning(f"Skipping line {line_num}: Invalid legacy format.")
                    continue

                parts = raw_text.split("<bot>:")
                if len(parts) > 2:
                    logging.warning(f"Line {line_num}: Multi-turn conversation detected with {len(parts)-1} bot turns. Using only the last turn.")

                # Extract the last assistant turn
                user_part = parts[0].replace("<human>:", "").strip()
                assistant_part = parts[-1].strip() if len(parts) > 1 else ""

                if not user_part or not assistant_part:
                    logging.warning(f"Skipping line {line_num}: Empty user or assistant content after parsing.")
                    continue
                
                # Construct ChatML Schema
                chatml_record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a highly intelligent, precise, and helpful personal assistant."
                        },
                        {
                            "role": "user",
                            "content": user_part
                        },
                        {
                            "role": "assistant",
                            "content": assistant_part
                        }
                    ]
                }
                
                outfile.write(json.dumps(chatml_record) + "\n")
                migrated_count += 1
                
            except Exception as e:
                logging.error(f"Error parsing line {line_num}: {str(e)}")
                
    logging.info(f"Migration Complete: {migrated_count} records converted and saved to {output_file}")

if __name__ == "__main__":
    # Define paths relative to the project root
    INPUT_PATH = "data/custom_data.jsonl"
    OUTPUT_PATH = "data/chatml_data.jsonl"
    
    migrate_legacy_data(INPUT_PATH, OUTPUT_PATH)