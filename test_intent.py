import asyncio
from agent.intent import classify_intent
from shared.schemas import ImageMetadata, Modality

imgs = [
    ImageMetadata(
        image_id="img1",
        modality=Modality.OPTICAL,
        crs=None,
        bounds=None,
        width=256,
        height=256,
        band_count=3,
        dtype="uint8",
        resolution_m=None,
        timestamp=None,
        cog_path="test.tif",
        is_valid=True,
    )
]

queries = [
    "Describe the land-cover and major objects in this image.",
    "Highlight the water body.",
    "Where is the water body?",
    "What objects are present in this image?",
]

async def main():
    for q in queries:
        result = await classify_intent(q, imgs)
        print(q, "=>", result)

asyncio.run(main())
