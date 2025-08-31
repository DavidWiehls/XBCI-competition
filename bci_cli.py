"""
BCI Motor Imagery Classifier Command Line Interface

A simple command-line interface for running offline BCI classification.
This provides an alternative to the GUI for users who prefer text-based interaction.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from offline_bci_classifier import OfflineBCIClassifier
except ImportError as e:
    print(f"Error: Could not import offline classifier: {e}")
    sys.exit(1)

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="BCI Motor Imagery Classifier - Offline Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and classify files in default directory
  python bci_cli.py --train-classify

  # Train and classify files in specific directory
  python bci_cli.py --train-classify --data-dir /path/to/data

  # Classify files using existing classifier
  python bci_cli.py --classify-only --data-dir /path/to/data

  # Load specific classifier and classify
  python bci_cli.py --load-classifier my_classifier.pkl --classify-only --data-dir /path/to/data

Class Labels:
  0: right_hand
  1: left_hand
  2: right_feet
  3: left_feet
        """
    )
    
    # Data directory
    parser.add_argument(
        "--data-dir", 
        default="off_line_data",
        help="Directory containing .npy data files (default: off_line_data)"
    )
    
    # Actions
    parser.add_argument(
        "--train-classify",
        action="store_true",
        help="Train a new classifier and classify all files"
    )
    
    parser.add_argument(
        "--classify-only",
        action="store_true", 
        help="Classify files using existing classifier"
    )
    
    parser.add_argument(
        "--load-classifier",
        help="Path to saved classifier file (.pkl)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        help="Save results to file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train_classify and not args.classify_only:
        parser.error("Must specify either --train-classify or --classify-only")
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        sys.exit(1)
    
    # Check for .npy files
    npy_files = list(Path(args.data_dir).glob("*.npy"))
    if not npy_files:
        print(f"Error: No .npy files found in '{args.data_dir}'")
        sys.exit(1)
    
    print(f"Found {len(npy_files)} .npy files in '{args.data_dir}'")
    
    # Initialize classifier
    classifier = OfflineBCIClassifier(sample_rate=500)
    
    try:
        if args.train_classify:
            print("\n" + "="*60)
            print("TRAINING CLASSIFIER")
            print("="*60)
            
            # Train classifier
            accuracy = classifier.train_classifier_from_files(args.data_dir)
            
            # Save classifier
            classifier.save_classifier("offline_bci_classifier.pkl")
            print(f"\nTraining completed with accuracy: {accuracy:.3f}")
            print("Classifier saved to 'offline_bci_classifier.pkl'")
            
        elif args.classify_only:
            # Load classifier
            if args.load_classifier:
                if not os.path.exists(args.load_classifier):
                    print(f"Error: Classifier file '{args.load_classifier}' does not exist")
                    sys.exit(1)
                classifier.load_classifier(args.load_classifier)
                print(f"Loaded classifier from '{args.load_classifier}'")
            else:
                # Try to load default classifier
                try:
                    classifier.load_classifier("offline_bci_classifier.pkl")
                    print("Loaded existing classifier from 'offline_bci_classifier.pkl'")
                except:
                    print("Error: No trained classifier found. Please train first using --train-classify")
                    sys.exit(1)
        
        # Classify files
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        
        results = classifier.classify_offline_directory(args.data_dir)
        
        print("\nSummary:")
        for filename, predicted_class in results:
            class_name = classifier.get_class_name(predicted_class)
            print(f"{filename}: Class {predicted_class} ({class_name})")
        
        # Show prediction probabilities for first file if verbose
        if args.verbose and npy_files:
            first_file = npy_files[0]
            import numpy as np
            data = np.load(first_file)
            probabilities = classifier.get_prediction_probabilities(data)
            
            print(f"\nPrediction probabilities for {first_file.name}:")
            for i, (class_name, prob) in enumerate(zip(classifier.classes, probabilities)):
                print(f"  Class {i} ({class_name}): {prob:.3f}")
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write("BCI Motor Imagery Classification Results\n")
                f.write("="*50 + "\n\n")
                f.write("Summary:\n")
                for filename, predicted_class in results:
                    class_name = classifier.get_class_name(predicted_class)
                    f.write(f"{filename}: Class {predicted_class} ({class_name})\n")
                
                if args.verbose and npy_files:
                    first_file = npy_files[0]
                    import numpy as np
                    data = np.load(first_file)
                    probabilities = classifier.get_prediction_probabilities(data)
                    
                    f.write(f"\nPrediction probabilities for {first_file.name}:\n")
                    for i, (class_name, prob) in enumerate(zip(classifier.classes, probabilities)):
                        f.write(f"  Class {i} ({class_name}): {prob:.3f}\n")
            
            print(f"\nResults saved to '{args.output}'")
        
        print("\nClassification completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
