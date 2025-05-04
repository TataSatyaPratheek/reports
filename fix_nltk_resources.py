#!/usr/bin/env python3
"""
Script to fix NLTK resource download and verification issues
"""
import nltk
import os
import shutil
import os

os.environ['NLTK_DATA'] = '/home/vi/nltk_data'
nltk.data.path.insert(0, '/home/vi/nltk_data')

def fix_nltk_resources():
    """Clear corrupted NLTK data and re-download"""
    
    # Get NLTK data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    
    print(f"NLTK data directory: {nltk_data_dir}")
    
    # Resources to fix
    resources = ['punkt', 'wordnet', 'stopwords']
    
    for resource in resources:
        resource_paths = [
            os.path.join(nltk_data_dir, 'tokenizers', resource),
            os.path.join(nltk_data_dir, 'corpora', resource),
            os.path.join(nltk_data_dir, 'tokenizers', f'{resource}.zip'),
            os.path.join(nltk_data_dir, 'corpora', f'{resource}.zip')
        ]
        
        # Remove existing files
        for path in resource_paths:
            if os.path.exists(path):
                print(f"Removing: {path}")
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        
        # Re-download
        print(f"Downloading {resource}...")
        try:
            nltk.download(resource, force=True, quiet=False)
            print(f"Successfully downloaded {resource}")
            
            # Verify
            try:
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif resource == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                print(f"Verified {resource}")
            except LookupError:
                print(f"Failed to verify {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    fix_nltk_resources()