import xlwings as xw
import time
import psutil
from PIL import ImageGrab

class ExcelTableScreenshot:
    def __init__(self):
        self.app = None
        self.workbook = None
    
    def close_existing_excel_processes(self):
        """Close any existing Excel processes to avoid conflicts"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and 'excel' in proc.info['name'].lower():
                    proc.terminate()
                    proc.wait(timeout=3)
            except:
                continue
    
    def open_excel_file(self, file_path: str) -> bool:
        """Open Excel file and prepare for screenshot"""
        try:
            # Clean slate - close existing Excel instances
            self.close_existing_excel_processes()
            time.sleep(1)
            
            # Open Excel with xlwings
            self.app = xw.App(visible=True)
            self.workbook = self.app.books.open(file_path)

            print(self.app.api.Name)
            print(self.app.api.Version)

            
            # Maximize window for optimal screenshot
            self.app.api.WindowState = -4137  # xlMaximized
            time.sleep(2)  # Allow Excel to fully load and render
            
            return True
        except Exception as e:
            print(f"Error opening Excel file: {e}")
            return False
    
    def capture_table_screenshot(self, output_path: str, sheet_name: str = None) -> bool:
        """Capture screenshot of the Excel table"""
        try:
            # Select the target sheet
            if sheet_name and sheet_name in [sheet.name for sheet in self.workbook.sheets]:
                target_sheet = self.workbook.sheets[sheet_name]
            else:
                target_sheet = self.workbook.sheets[0]  # First sheet by default
            
            target_sheet.activate()
            time.sleep(1)
            
            # Find and select the data range
            used_range = target_sheet.used_range
            if not used_range:
                print("No data found in the selected sheet")
                return False
            
            used_range.api.CopyPicture(Appearance = 1, Format = 2)

            # Capture the screenshot
            image = ImageGrab.grabclipboard()
            if image is not None:
                image.save(output_path)
            else:
                print("Clipboard is empty")
            
            return True
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return False
    
    def close_excel(self):
        """Clean up Excel resources"""
        try:
            if self.workbook:
                self.workbook.close()
            if self.app:
                self.app.quit()
        except Exception as e:
            print(f"Error closing Excel: {e}")
    
    def take_screenshot(self, excel_file_path: str, output_image_path: str, sheet_name: str = None) -> bool:
        """Complete workflow: open Excel, capture screenshot, cleanup"""
        try:
            # Open Excel file
            if not self.open_excel_file(excel_file_path):
                return False
            
            # Capture screenshot
            success = self.capture_table_screenshot(output_image_path, sheet_name)
            
            return success
            
        except Exception as e:
            print(f"Screenshot process failed: {e}")
            return False
        finally:
            # Always cleanup, even if there's an error
            self.close_excel()