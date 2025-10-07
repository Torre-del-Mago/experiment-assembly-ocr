import torch
import os

def update_craft_checkpoint():
    # File paths
    old_checkpoint_path = "CRAFT_clr_amp_29500.pth"
    new_craft_path = "craft_mlt_25k.pth"
    output_path = "starting_CRAFT.pth"
    
    # Check if files exist
    if not os.path.exists(old_checkpoint_path):
        print(f"Error: File not found {old_checkpoint_path}")
        return
    
    if not os.path.exists(new_craft_path):
        print(f"Error: File not found {new_craft_path}")
        return
    
    try:
        # Load old checkpoint with additional keys
        print("Loading old checkpoint...")
        old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
        print(f"Keys in old checkpoint: {list(old_checkpoint.keys())}")
        
        # Load new CRAFT model
        print("Loading new CRAFT model...")
        new_craft = torch.load(new_craft_path, map_location='cpu')
        print(f"Type of new CRAFT: {type(new_craft)}")
        
        # If new_craft is already a model state_dict, use it directly
        if isinstance(new_craft, dict) and any(key.startswith(('module.', 'basenet.', 'upconv')) for key in new_craft.keys()):
            new_craft_state_dict = new_craft
        else:
            # If it's a checkpoint with 'craft' key or similar
            if 'craft' in new_craft:
                new_craft_state_dict = new_craft['craft']
            elif 'model' in new_craft:
                new_craft_state_dict = new_craft['model']
            elif 'state_dict' in new_craft:
                new_craft_state_dict = new_craft['state_dict']
            else:
                new_craft_state_dict = new_craft
        
        print(f"Keys in new CRAFT (first 5): {list(new_craft_state_dict.keys())[:5]}")
        
        # Create new checkpoint
        new_checkpoint = {}
        
        # Copy all keys except 'craft'
        for key, value in old_checkpoint.items():
            if key != 'craft':
                new_checkpoint[key] = value
                print(f"Copied key: {key}")
        
        # Add new CRAFT model
        new_checkpoint['craft'] = new_craft_state_dict
        print("Added new CRAFT model")
        
        # Save new checkpoint
        print(f"Saving new checkpoint to {output_path}...")
        torch.save(new_checkpoint, output_path)
        print("Checkpoint saved successfully!")
        
        # Display summary
        print(f"\nSummary:")
        print(f"- Old checkpoint: {old_checkpoint_path}")
        print(f"- New CRAFT model: {new_craft_path}")
        print(f"- Output checkpoint: {output_path}")
        print(f"- Keys in final checkpoint: {list(new_checkpoint.keys())}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    update_craft_checkpoint()