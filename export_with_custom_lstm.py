import torch
import argparse

# Import your model definitions
from model_def import CausalDemucsSplit, FRAME_LENGTH, CustomLSTM  # adjust import path as needed


def replace_lstm_with_custom(model: CausalDemucsSplit) -> None:
    """
    Replace the PyTorch nn.LSTM in the Demucs model with the custom LSTM,
    copying over all weights and biases.
    """
    orig_lstm = model.lstm
    # Create custom LSTM with same configuration
    custom = CustomLSTM(
        input_size=orig_lstm.input_size,
        hidden_size=orig_lstm.hidden_size,
        num_layers=orig_lstm.num_layers,
        bias=orig_lstm.bias,
        batch_first=orig_lstm.batch_first,
        bidirectional=orig_lstm.bidirectional
    )
    # Transfer weights and biases
    for name, param in orig_lstm.named_parameters():
        # custom parameters use the same names
        cp = getattr(custom, name)
        cp.data.copy_(param.data)
    # Replace in model
    model.lstm = custom


def export_to_onnx(model: torch.nn.Module, output_path: str):
    """
    Export the full end-to-end model to ONNX (opset 9), with fixed input shape.
    """
    model.eval()
    # ONNX expects (batch, channels, length)
    dummy_input = torch.randn(1, 1, FRAME_LENGTH)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=9,
        input_names=["audio"],
        output_names=["enhanced"],
        dynamic_axes=None,
        verbose=False
    )
    print(f"Model exported to {output_path} (opset 9)")


def main():
    parser = argparse.ArgumentParser(description="Export Demucs model with custom LSTM to ONNX opset 9.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--output", type=str, default="demucs.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    # Initialize model (use same hyperparameters as training)
    model = CausalDemucsSplit()
    # Load PyTorch checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # If checkpoint is a state_dict directly, use it; else expect {'model_state_dict': ...}
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)

    # Replace LSTM
    replace_lstm_with_custom(model)
    # Export
    export_to_onnx(model, args.output)


if __name__ == "__main__":
    main()