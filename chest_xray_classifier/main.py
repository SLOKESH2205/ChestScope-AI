"""
Command-line interface for chest X-ray classification pipeline.

Provides subcommands for training, evaluation, prediction, explanation,
report generation, and web interface.

Usage:
    python main.py train --model custom_cnn
    python main.py predict --image path/to/image.jpg
    python main.py evaluate --model custom_cnn
    python main.py app
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
import time
import os
import sys

# Allow running main.py directly from inside the package folder.
if __package__ is None:
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import numpy as np
from tensorflow import keras

from chest_xray_classifier.config.config import (
    MODEL_PATH, EFFNET_PATH, LOG_DIR, SEED, BATCH_SIZE, EPOCHS_CNN,
    LR_CNN, EPOCHS_EFFNET_PHASE1, EPOCHS_EFFNET_PHASE2, LR_EFFNET_PHASE1,
    LR_EFFNET_PHASE2, MOBILENET_PATH, EPOCHS_MOBILENET_PHASE1,
    EPOCHS_MOBILENET_PHASE2, LR_MOBILENET_PHASE1, LR_MOBILENET_PHASE2
)
from chest_xray_classifier.data.loader import DataLoader
from chest_xray_classifier.train.trainer import Trainer
from chest_xray_classifier.evaluate.evaluator import Evaluator
from chest_xray_classifier.predict import run_single_inference, CONFIDENCE_THRESHOLD

class Predictor:
    def __init__(self, model, enable_uncertainty=True):
        self.model = model
        self.enable_uncertainty = enable_uncertainty
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def predict_single(self, image_path, include_uncertainty=True):
        res = run_single_inference(
            self.model,
            image_path,
            model_name="Custom CNN",
            confidence_threshold=self.confidence_threshold
        )
        res["image_path"] = str(image_path)
        return res

    def format_to_ascii_card(self, res):
        lines = []
        lines.append("+" + "-"*65 + "+")
        lines.append(f"| CHEST X-RAY AI CLINICAL PREDICTION CARD{' ':24}|")
        lines.append("+" + "-"*65 + "+")
        lines.append(f"| Filename: {res['filename']:52} |")
        lines.append(f"| Model: {res['model_name']:55} |")
        lines.append(f"| Prediction: {res['prediction']:50} |")
        lines.append(f"| Confidence: {res['confidence']:.2%} ({'Acceptable' if not res['requires_review'] else 'Clinical Review Required'}){' ':8} |")
        lines.append(f"| Uncertainty (Entropy): {res['uncertainty']:.2%}{' ':35} |")
        lines.append(f"| Inference Latency: {res['inference_ms']:.1f} ms{' ':36} |")
        lines.append("+" + "-"*65 + "+")
        lines.append("| Probabilities:                                                  |")
        for cls, prob in res['probabilities'].items():
            bar_len = int(prob * 20)
            bar = "#" * bar_len + " " * (20 - bar_len)
            lines.append(f"|   {cls:25}: {prob:7.2%} [{bar}] {' ':12} |")
        lines.append("+" + "-"*65 + "+")
        return "\n".join(lines)


class PredictionLogger:
    def log_prediction(self, res, model_name='cnn', inference_time_ms=None):
        from chest_xray_classifier.utils.portfolio import save_prediction_log
        save_prediction_log(
            model_name=res.get("model_name", model_name),
            filename=res["filename"],
            prediction=res["prediction"],
            confidence=res["confidence"],
            uncertainty=res["uncertainty"],
            inference_ms=inference_time_ms or res["inference_ms"]
        )
        return "Saved"

    def get_statistics(self):
        from chest_xray_classifier.utils.portfolio import load_prediction_history
        df = load_prediction_history()
        if df.empty:
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "class_distribution": {}
            }
        class_counts = df['prediction'].value_counts().to_dict()
        return {
            "total_predictions": len(df),
            "avg_confidence": float(df['confidence'].mean()),
            "class_distribution": class_counts
        }

    def export_csv(self, export_path):
        from chest_xray_classifier.utils.portfolio import load_prediction_history
        df = load_prediction_history()
        df.to_csv(export_path, index=False)
        return export_path
from chest_xray_classifier.explain.gradcam import GradCAM
from chest_xray_classifier.explain.shap_explain import SHAPExplainer
from chest_xray_classifier.explain.uncertainty import MCDropoutUncertainty
from chest_xray_classifier.visualize.plots import (
    plot_training_history, plot_confusion_matrix, plot_roc_curves
)
def generate_simple_report(res):
    from chest_xray_classifier.app.report.pdf_generator import generate_clinical_pdf
    pdf_bytes = generate_clinical_pdf(
        image_path=res["image_path"],
        gradcam_path=None,
        prediction=res["prediction"],
        confidence=res["confidence"],
        uncertainty=res["uncertainty"],
        model_name=res["model_name"],
        inference_ms=res["inference_ms"],
        threshold=CONFIDENCE_THRESHOLD
    )
    report_path = Path("outputs") / f"report_{Path(res['image_path']).stem}.pdf"
    report_path.write_bytes(pdf_bytes)
    return report_path

# Setup logging
log_dir = Path(LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"chest_xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_model_path(model_name: str) -> Path:
    """Return the saved model path for a given model key."""
    if model_name == 'efficientnet':
        return EFFNET_PATH
    if model_name == 'mobilenetv2':
        return MOBILENET_PATH
    return MODEL_PATH


def train_command(args):
    """Train model command."""
    logger.info("="*80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load data
    data_loader = DataLoader()
    if not data_loader.verify_dataset_structure():
        logger.error("Dataset structure is invalid. Please check your data directory.")
        return
    
    logger.info("Loading data generators...")
    train_gen = data_loader.get_train_generator()
    val_gen = data_loader.get_val_generator()
    
    # Display dataset info
    logger.info("\nDataset Statistics:")
    class_dist = data_loader.get_class_distribution()
    logger.info(f"\n{class_dist.to_string(index=False)}")
    
    # Train
    trainer = Trainer(experiment_name=f"chest_xray_{args.model}")
    
    if args.model == 'custom_cnn':
        logger.info("\nTraining Custom CNN...")
        model, history = trainer.train_custom_cnn(
            train_gen,
            val_gen,
            epochs=EPOCHS_CNN,
            learning_rate=LR_CNN,
            dropout_rate=0.4,
            l2_factor=0.001
        )
    
    elif args.model == 'efficientnet':
        logger.info("\nTraining EfficientNetB0...")
        model, history = trainer.train_efficientnet(
            train_gen,
            val_gen,
            phase1_epochs=EPOCHS_EFFNET_PHASE1,
            phase2_epochs=EPOCHS_EFFNET_PHASE2,
            phase1_lr=LR_EFFNET_PHASE1,
            phase2_lr=LR_EFFNET_PHASE2
        )

    elif args.model == 'mobilenetv2':
        logger.info("\nTraining MobileNetV2...")
        model, history = trainer.train_mobilenetv2(
            train_gen,
            val_gen,
            phase1_epochs=EPOCHS_MOBILENET_PHASE1,
            phase2_epochs=EPOCHS_MOBILENET_PHASE2,
            phase1_lr=LR_MOBILENET_PHASE1,
            phase2_lr=LR_MOBILENET_PHASE2
        )
    
    else:
        logger.error(f"Unknown model: {args.model}")
        return
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)


def evaluate_command(args):
    """Evaluate model command."""
    logger.info("="*80)
    logger.info("STARTING EVALUATION")
    logger.info("="*80)
    
    # Load model
    model_path = get_model_path(args.model)
    try:
        model = keras.models.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model not found at {model_path}")
        return
    
    # Load data
    data_loader = DataLoader()
    val_gen = data_loader.get_val_generator()
    
    # Evaluate
    evaluator = Evaluator(model, model_name=args.model)
    results = evaluator.evaluate_on_generator(val_gen)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    if args.save:
        evaluator.save_results(results)
    
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


def predict_command(args):
    """Single image prediction command."""
    logger.info("="*80)
    logger.info("RUNNING PREDICTION")
    logger.info("="*80)
    
    # Load model
    try:
        model = keras.models.load_model(str(MODEL_PATH))
        logger.info(f"Model loaded")
    except FileNotFoundError:
        logger.error(f"Model not found")
        return
    
    # Initialize predictor
    predictor = Predictor(model, enable_uncertainty=True)
    
    # Predict
    logger.info(f"Predicting on: {args.image}")
    start_time = time.time()
    
    prediction = predictor.predict_single(args.image, include_uncertainty=True)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Print result
    print("\n" + predictor.format_to_ascii_card(prediction))
    logger.info(f"Inference time: {inference_time:.2f} ms")
    
    # Log to database
    if args.log:
        pred_logger = PredictionLogger()
        pred_id = pred_logger.log_prediction(prediction, model_name='cnn', inference_time_ms=inference_time)
        logger.info(f"Prediction logged (ID: {pred_id})")
    
    # Generate report
    if args.report:
        report_path = generate_simple_report(prediction)
        logger.info(f"Report saved to {report_path}")
    
    logger.info("="*80)


def explain_command(args):
    """Generate explainability visualizations."""
    logger.info("="*80)
    logger.info("GENERATING EXPLANATIONS")
    logger.info("="*80)
    
    # Load model
    try:
        model = keras.models.load_model(str(MODEL_PATH))
    except FileNotFoundError:
        logger.error("Model not found")
        return
    
    # Load image
    data_loader = DataLoader()
    image = data_loader.load_single_image(args.image)
    
    if args.method == 'gradcam' or args.method == 'all':
        logger.info("Generating Grad-CAM explanation...")
        gradcam = GradCAM(model)
        result = gradcam.explain_prediction(image)
        logger.info(f"Grad-CAM heatmap generated")
    
    if args.method == 'uncertainty' or args.method == 'all':
        logger.info("Computing MC Dropout uncertainty...")
        uncertainty_estimator = MCDropoutUncertainty(model, n_iterations=50)
        uncertainty_result = uncertainty_estimator.predict_with_uncertainty(image)
        logger.info(f"Total Uncertainty: {uncertainty_result['total_uncertainty']:.4f}")
    
    logger.info("="*80)


def history_command(args):
    """View prediction history."""
    logger.info("="*80)
    logger.info("PREDICTION HISTORY")
    logger.info("="*80)
    
    pred_logger = PredictionLogger()
    
    # Get statistics
    stats = pred_logger.get_statistics()
    
    logger.info(f"\nTotal Predictions: {stats.get('total_predictions', 0)}")
    logger.info(f"Avg Confidence: {stats.get('avg_confidence', 0):.2%}")
    
    if 'accuracy' in stats:
        logger.info(f"Accuracy: {stats['accuracy']:.4f}")
    
    logger.info(f"\nClass Distribution:")
    for class_name, count in stats.get('class_distribution', {}).items():
        logger.info(f"  {class_name}: {count}")
    
    # Export if requested
    if args.export:
        csv_path = pred_logger.export_csv(args.export)
        logger.info(f"Exported to {csv_path}")
    
    logger.info("="*80)


def app_command(args):
    """Launch Streamlit web interface."""
    logger.info("="*80)
    logger.info("LAUNCHING STREAMLIT APP")
    logger.info("="*80)
    
    import subprocess
    import sys
    
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        logger.error(f"App not found at {app_path}")
        return
    
    logger.info(f"Starting Streamlit app at {app_path}")
    logger.info("Open your browser to the provided URL (usually http://localhost:8501)")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --model custom_cnn
  python main.py evaluate --model custom_cnn
  python main.py predict --image path/to/xray.jpg
  python main.py explain --method gradcam --image path/to/xray.jpg
  python main.py history --export predictions.csv
  python main.py app
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--model',
        choices=['custom_cnn', 'efficientnet', 'mobilenetv2'],
        default='custom_cnn',
        help='Model architecture to train'
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument(
        '--model',
        choices=['custom_cnn', 'efficientnet', 'mobilenetv2'],
        default='custom_cnn',
        help='Model to evaluate'
    )
    eval_parser.add_argument('--save', action='store_true', help='Save results')
    eval_parser.set_defaults(func=evaluate_command)
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict on image')
    pred_parser.add_argument('--image', required=True, help='Path to image')
    pred_parser.add_argument('--log', action='store_true', help='Log to database')
    pred_parser.add_argument('--report', action='store_true', help='Generate report')
    pred_parser.set_defaults(func=predict_command)
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain prediction')
    explain_parser.add_argument('--image', required=True, help='Path to image')
    explain_parser.add_argument(
        '--method',
        choices=['gradcam', 'uncertainty', 'all'],
        default='all',
        help='Explanation method'
    )
    explain_parser.set_defaults(func=explain_command)
    
    # History command
    hist_parser = subparsers.add_parser('history', help='View prediction history')
    hist_parser.add_argument(
        '--export',
        help='Export to CSV file'
    )
    hist_parser.set_defaults(func=history_command)
    
    # App command
    app_parser = subparsers.add_parser('app', help='Launch web interface')
    app_parser.set_defaults(func=app_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run command
    args.func(args)


if __name__ == '__main__':
    main()
