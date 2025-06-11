import os
import datetime

def analyze_directory(directory="."):
    """
    Analyzes the given directory and generates a summary report.

    Args:
        directory (str): The path to the directory to analyze.

    Returns:
        str: A formatted string containing the project summary report.
    """
    report = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report.append("="*50)
    report.append(f"Project Analysis Report")
    report.append(f"Generated on: {now}")
    report.append(f"Directory: {os.path.abspath(directory)}")
    report.append("="*50)
    
    try:
        items = os.listdir(directory)
        file_count = 0
        dir_count = 0
        file_types = {}

        for item in items:
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                dir_count += 1
            elif os.path.isfile(path):
                file_count += 1
                ext = os.path.splitext(item)[1]
                if ext == "":
                    ext = "No Extension"
                file_types[ext] = file_types.get(ext, 0) + 1
        
        report.append(f"\n--- Overview ---")
        report.append(f"Total Items: {len(items)}")
        report.append(f"Directories: {dir_count}")
        report.append(f"Files: {file_count}")
        
        if file_count > 0:
            report.append(f"\n--- File Types ---")
            sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)
            for ext, count in sorted_types:
                report.append(f"- {ext if ext else 'N/A'}: {count}")

    except FileNotFoundError:
        report.append("\nERROR: Directory not found.")
    except Exception as e:
        report.append(f"\nAn unexpected error occurred: {e}")
        
    report.append("\n" + "="*50)
    return "\n".join(report)

if __name__ == "__main__":
    report = analyze_directory()
    print(report)
    
    # Save the report to a file
    with open("project_analysis_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to project_analysis_report.txt") 