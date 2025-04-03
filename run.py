# Carrier and Farag Demographic Smoothing Technique
# Main entry point for running the demographic analysis and dashboard

import argparse
import os
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Carrier and Farag Demographic Smoothing Analysis')
    parser.add_argument('--dashboard', action='store_true', help='Run the interactive dashboard')
    parser.add_argument('--analysis', action='store_true', help='Run the demographic analysis')
    parser.add_argument('--year', type=int, default=2020, help='Year to analyze (default: 2020)')
    parser.add_argument('--sex', type=str, default='Both sexes', 
                        choices=['Both sexes', 'Male', 'Female'], 
                        help='Sex category to analyze (default: Both sexes)')
    parser.add_argument('--output', type=str, default='./output', help='Output directory for results (default: ./output)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # If no specific mode is selected, show help
    if not (args.dashboard or args.analysis):
        parser.print_help()
        print("\nPlease specify either --dashboard or --analysis to run the respective component.")
        return
    
    # Run the selected component
    if args.analysis:
        print("Running demographic analysis...")
        from main import main as run_analysis
        run_analysis(year=args.year, sex=args.sex, output_dir=args.output)
    
    if args.dashboard:
        print("Starting the interactive dashboard...")
        print("Open your web browser and navigate to http://127.0.0.1:8050/")
        import dashboard
        dashboard.app.run_server(debug=True)

if __name__ == "__main__":
    main()