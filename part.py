from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display

file_path = "/content/pdfs/1706.03762v7.pdf"

chunks = partition_pdf(
    filename=file_path, # path to the PDF file to be partitioned
    infer_table_structure=True, #enables automatic detection and structuring of tables within the document
    strategy="hi_res", #most accurate, but potentially slowest and most resource-intensive, strategy for analyzing a document's layout and content
    extract_image_block_types=["Image", "Tables"], #extracts images and tables locally
    extract_image_block_output_dir="images", #saves images and tables to the specified directory
    extract_image_block_to_payload=True, #metadata with base64
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)

elements = chunks[3].metadata.orig_elements
chunk_images = [el for el in elements if 'Image' in str(type(el))]
print(chunk_images[0].to_dict())

tables = []
texts= []
for chunk in chunks:
  if 'Table' in str(type(chunk)):
    tables.append(chunk)
  if 'CompositeElement' in str(type(chunk)):
    texts.append(chunk)

def get_images_base64(chunks):
  image_64=[]
  for chunk in chunks:
    if 'CompositeElement' in str(type(chunk)):
      chunk_el =chunk.metadata.orig_elements
      for el in chunk_el:
        if 'Image' in str(type(el)):
          image_64.append(el.metadata.image_base64)
  return image_64
images = get_images_base64(chunks)


def display_image_base64(base64_code):
  image_data= base64.b64decode(base64_code) #decode base64 to binary
  display(Image(data=image_data))

print(display_image_base64(images[2]))