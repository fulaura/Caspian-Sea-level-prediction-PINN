import os
import sys
import torch
from torchview import draw_graph

# Import the PINN-1 model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pinn1_water_balance import WaterBalancePINN1

def main():
    # 1. Initialize PINN-1 model
    model = WaterBalancePINN1(in_channels=13, seq_len=6)
    
    # 2. Define output path in results/pinn1
    output_dir = "antigravity/results/pinn1"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_architecture_pinn1")
    
    print("Generating HIGH-RESOLUTION LAYER-WISE architecture graph for PINN-1...")
    
    # 3. Use torchview with a Wrapper to handle multiple outputs cleanly
    class SingleOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # PINN-1 returns: (dh_pred, alpha_E, beta_volga, gamma_other, log_var_data, log_var_phys)
            return self.model(x)[0] 
            
    wrapped_model = SingleOutputWrapper(model)
    
    try:
        model_graph = draw_graph(
            wrapped_model, 
            input_size=(1, 6, 13, 110, 90),
            expand_nested=True,
            depth=3, # Shows named layers inside modules
            device='cpu',
            save_graph=False,
            hide_module_functions=True, # Hides tensor operations for cleaner look
            hide_inner_tensors=True,
        )
        
        # 4. Render with High Resolution
        print("Rendering PNG (300 DPI) with named layers...")
        model_graph.visual_graph.attr(dpi='300')
        model_graph.visual_graph.render(output_path, format="png", cleanup=True)
        
        print(f"Success! High-resolution PINN-1 architecture saved to {output_path}.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: If rendering fails, ensure Graphviz binaries are in your PATH.")

    # 5. Summary
    print("\nLayer-by-Layer Summary for PINN-1:")
    print(model)

if __name__ == "__main__":
    main()
