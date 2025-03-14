
import fitz  # PyMuPDF

class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.elements = []
        self.chunks = []
    
    def extract_text_with_styles(self):
        """Extract text with font sizes and styles from the PDF."""
        doc = fitz.open(self.pdf_path)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            self.elements.append({
                                "text": span["text"].strip(),
                                "size": span["size"],  # Font size
                                "bold": "Bold" in span["font"],  # Check if bold
                                "italic": "Italic" in span["font"]  # Check if italic
                            })
    
    def group_headings_with_content(self):
        """Group headings with their following content."""
        if not self.elements:
            return

        font_sizes = sorted(set(e["size"] for e in self.elements), reverse=True)
        heading_threshold = font_sizes[:2] if len(font_sizes) > 1 else font_sizes[:1]

        current_heading = None
        current_content = []

        for element in self.elements:
            is_heading = element["size"] in heading_threshold
            if is_heading:
                if current_heading and current_content:
                    self.chunks.append({"title": current_heading, "content": " ".join(current_content)})
                current_heading = element["text"]
                current_content = []
            else:
                if element["text"]:
                    current_content.append(element["text"])
        
        if current_heading and current_content:
            self.chunks.append({"title": current_heading, "content": " ".join(current_content)})
    
    def process_pdf(self):
        """Extract and process PDF text into structured chunks."""
        self.extract_text_with_styles()
        self.group_headings_with_content()
        return self.chunks

# if __name__ == "__main__":
#     pdf_path = "dataset/Serri_doc.pdf"  # Update with actual file path
#     extractor = PDFTextExtractor(pdf_path)
#     chunks = extractor.process_pdf()
    
#     for chunk in chunks:
#         print(f"\nTitle: {chunk['title']}")
#         print(f"Content: {chunk['content'][:500]}...")
#         print("-" * 80)
