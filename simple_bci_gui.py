"""
Simple BCI Motor Imagery Classifier GUI

A user-friendly graphical interface for running both online and offline BCI classification.
This simplified version focuses on ease of use and reliability.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import numpy as np

# Try to import PIL for image handling
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Logo will not be displayed.")

# Import our BCI classifiers
try:
    from offline_bci_classifier import OfflineBCIClassifier
except ImportError as e:
    print(f"Warning: Could not import offline classifier: {e}")

class SimpleBCIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XBCI Motor Imagery Classifier")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize classifier
        self.offline_classifier = None
        
        # Load logo
        self.load_logo()
        
        # Create GUI elements
        self.create_widgets()
        
    def load_logo(self):
        """Load and resize the XBCI logo"""
        self.logo_image = None
        
        if not PIL_AVAILABLE:
            print("PIL not available, skipping logo loading")
            return
            
        try:
            logo_path = "assets_ui/xbci_logo.PNG"
            if os.path.exists(logo_path):
                # Load image
                image = Image.open(logo_path)
                # Resize to appropriate size for GUI
                image = image.resize((200, 80), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(image)
                print("XBCI logo loaded successfully")
            else:
                print(f"Logo not found at {logo_path}")
        except Exception as e:
            print(f"Error loading logo: {e}")
        
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header frame with logo and title
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Logo (left side)
        if self.logo_image:
            logo_label = ttk.Label(header_frame, image=self.logo_image)
            logo_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Title (right side)
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        title_label = ttk.Label(title_frame, text="XBCI Motor Imagery Classifier", 
                               font=("Arial", 18, "bold"))
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(title_frame, text="Brain-Computer Interface Classification System", 
                                  font=("Arial", 10))
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Classification Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="offline")
        
        ttk.Radiobutton(mode_frame, text="Offline Classification (Process saved data files)", 
                       variable=self.mode_var, value="offline").pack(anchor=tk.W)
        
        ttk.Radiobutton(mode_frame, text="Online Classification (Real-time from device)", 
                       variable=self.mode_var, value="online").pack(anchor=tk.W)
        
        # Offline controls
        self.offline_frame = ttk.LabelFrame(main_frame, text="Offline Classification", padding="10")
        self.offline_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Data directory selection
        dir_frame = ttk.Frame(self.offline_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dir_frame, text="Data Directory:").pack(anchor=tk.W)
        
        dir_input_frame = ttk.Frame(dir_frame)
        dir_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.dir_var = tk.StringVar(value="off_line_data")
        self.dir_entry = ttk.Entry(dir_input_frame, textvariable=self.dir_var)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(dir_input_frame, text="Browse", 
                  command=self.browse_directory).pack(side=tk.RIGHT)
        
        # Offline buttons
        button_frame = ttk.Frame(self.offline_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Train & Classify", 
                  command=self.train_and_classify).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Load Saved Classifier", 
                  command=self.load_classifier).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Classify Only", 
                  command=self.classify_only).pack(side=tk.LEFT)
        
        # Online controls
        self.online_frame = ttk.LabelFrame(main_frame, text="Online Classification", padding="10")
        self.online_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Online info
        info_text = """
Online classification requires:
1. A BLE-compatible EEG device
2. The device to be paired and connected
3. Running the online classifier script separately

For online classification, please run:
python bci_motor_imagery_classifier_v2.py
        """
        
        ttk.Label(self.online_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(results_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(control_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Save Results", 
                  command=self.save_results).pack(side=tk.LEFT, padx=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select offline mode and choose data directory")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Initially show offline controls
        self.show_offline_controls()
        
    def show_offline_controls(self):
        """Show offline controls and hide online"""
        self.offline_frame.pack(fill=tk.X, pady=(0, 10))
        self.online_frame.pack_forget()
        
    def show_online_controls(self):
        """Show online controls and hide offline"""
        self.offline_frame.pack_forget()
        self.online_frame.pack(fill=tk.X, pady=(0, 10))
        
    def log_message(self, message):
        """Add message to results area"""
        self.results_text.insert(tk.END, str(message) + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
        
    def browse_directory(self):
        """Browse for data directory"""
        directory = filedialog.askdirectory(initialdir=self.dir_var.get())
        if directory:
            self.dir_var.set(directory)
            self.log_message(f"Selected directory: {directory}")
            
    def train_and_classify(self):
        """Train classifier and classify files"""
        def train_and_classify_thread():
            try:
                data_dir = self.dir_var.get()
                if not os.path.exists(data_dir):
                    messagebox.showerror("Error", f"Directory {data_dir} does not exist")
                    return
                    
                self.status_var.set("Training classifier...")
                self.log_message("=" * 60)
                self.log_message("TRAINING OFFLINE CLASSIFIER")
                self.log_message("=" * 60)
                
                # Initialize classifier
                self.offline_classifier = OfflineBCIClassifier(sample_rate=500)
                
                # Train classifier
                accuracy = self.offline_classifier.train_classifier_from_files(data_dir)
                
                # Save classifier
                self.offline_classifier.save_classifier("offline_bci_classifier.pkl")
                
                self.log_message(f"Training completed with accuracy: {accuracy:.3f}")
                self.log_message("Classifier saved to 'offline_bci_classifier.pkl'")
                
                # Now classify files
                self.log_message("\n" + "=" * 60)
                self.log_message("CLASSIFICATION RESULTS")
                self.log_message("=" * 60)
                
                results = self.offline_classifier.classify_offline_directory(data_dir)
                
                self.log_message("\nSummary:")
                for filename, predicted_class in results:
                    class_name = self.offline_classifier.get_class_name(predicted_class)
                    self.log_message(f"{filename}: Class {predicted_class} ({class_name})")
                
                # Show prediction probabilities for first file
                files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
                if files:
                    files.sort()
                    first_file = os.path.join(data_dir, files[0])
                    data = np.load(first_file)
                    probabilities = self.offline_classifier.get_prediction_probabilities(data)
                    
                    self.log_message(f"\nPrediction probabilities for {files[0]}:")
                    for i, (class_name, prob) in enumerate(zip(self.offline_classifier.classes, probabilities)):
                        self.log_message(f"  Class {i} ({class_name}): {prob:.3f}")
                
                self.status_var.set("Training and classification completed")
                
            except Exception as e:
                error_msg = f"Error: {e}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
                self.status_var.set("Error occurred")
                
        # Run in separate thread
        threading.Thread(target=train_and_classify_thread, daemon=True).start()
        
    def classify_only(self):
        """Classify files using existing classifier"""
        def classify_thread():
            try:
                # Try to load existing classifier
                if self.offline_classifier is None:
                    try:
                        self.offline_classifier = OfflineBCIClassifier()
                        self.offline_classifier.load_classifier("offline_bci_classifier.pkl")
                        self.log_message("Loaded existing classifier from 'offline_bci_classifier.pkl'")
                    except:
                        messagebox.showerror("Error", "No trained classifier found. Please train first.")
                        return
                
                data_dir = self.dir_var.get()
                if not os.path.exists(data_dir):
                    messagebox.showerror("Error", f"Directory {data_dir} does not exist")
                    return
                    
                self.status_var.set("Classifying files...")
                self.log_message("=" * 60)
                self.log_message("CLASSIFICATION RESULTS")
                self.log_message("=" * 60)
                
                results = self.offline_classifier.classify_offline_directory(data_dir)
                
                self.log_message("\nSummary:")
                for filename, predicted_class in results:
                    class_name = self.offline_classifier.get_class_name(predicted_class)
                    self.log_message(f"{filename}: Class {predicted_class} ({class_name})")
                
                self.status_var.set("Classification completed")
                
            except Exception as e:
                error_msg = f"Error: {e}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
                self.status_var.set("Error occurred")
                
        # Run in separate thread
        threading.Thread(target=classify_thread, daemon=True).start()
        
    def load_classifier(self):
        """Load saved classifier"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Classifier File",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if file_path:
                self.offline_classifier = OfflineBCIClassifier()
                self.offline_classifier.load_classifier(file_path)
                self.log_message(f"Loaded classifier from {file_path}")
                self.status_var.set("Classifier loaded successfully")
                
        except Exception as e:
            error_msg = f"Failed to load classifier: {e}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        
    def save_results(self):
        """Save results to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                self.log_message(f"Results saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SimpleBCIGUI(root)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
