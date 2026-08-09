import os
import sys
import torch
from torchview import draw_graph

# Import the model from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pinn2_budyko import BudykoPINN2_v21

def main():
    # 1. Initialize model
    model = BudykoPINN2_v21(in_channels=13, seq_len=6)
    
    # 2. Define output path
    output_path = "antigravity/results/pinn2/model_architecture_v2"
    
    print("Generating HIGH-RESOLUTION LAYER-WISE architecture graph...")
    
    # 3. Use torchview with a Wrapper to handle multiple outputs
    # torchview sometimes struggles with tuple outputs; wrapping ensures a single graph root
    class SingleOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)[0] # Only dh_pred
            
    wrapped_model = SingleOutputWrapper(model)
    
    try:
        model_graph = draw_graph(
            wrapped_model, 
            input_size=(1, 6, 13, 110, 90),
            expand_nested=True,
            depth=3, # Increased depth to show named layers inside modules
            device='cpu',
            save_graph=False,
            hide_module_functions=True, # Hides 'view', 'getitem', etc. for cleaner look
            hide_inner_tensors=True,
        )
        
        # 4. Render with High Resolution
        print("Rendering PNG (300 DPI) with named layers...")
        model_graph.visual_graph.attr(dpi='300')
        model_graph.visual_graph.render(output_path, format="png", cleanup=True)
        
        print(f"Success! High-resolution architecture saved to {output_path}.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: If rendering fails, ensure Graphviz binaries are in your PATH.")

    # 5. Summary
    print("\nLayer-by-Layer Summary:")
    print(model)

if __name__ == "__main__":
    main()
