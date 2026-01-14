"""
L2: Vision/multi-modal tool using Gemini Vision API.
"""

import base64
from pathlib import Path
from typing import Optional, Union
import logging

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class VisionTool(BaseTool):
    """
    L2: Vision/multi-modal tool.

    Provides image analysis capabilities using Gemini Vision API:
    - Image description
    - Chart/graph analysis
    - OCR-like text extraction
    - Visual question answering
    """

    name: str = "vision"
    level: int = 2

    def __init__(self, llm_client=None):
        """
        Initialize Vision tool.

        Args:
            llm_client: LLM client with vision capabilities
        """
        self.llm_client = llm_client

    def _load_image(self, image_source: Union[str, bytes, Path]) -> Optional[bytes]:
        """
        Load image from various sources.

        Args:
            image_source: File path, URL, or raw bytes

        Returns:
            Image bytes or None if failed
        """
        if isinstance(image_source, bytes):
            return image_source

        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.exists():
                return path.read_bytes()

            # Could be a URL - would need to fetch
            logger.warning(f"Image path not found: {image_source}")
            return None

        return None

    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image to base64 for API."""
        return base64.b64encode(image_bytes).decode('utf-8')

    def execute(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> ToolResult:
        """
        Execute vision analysis.

        Args:
            query: Question about the image
            context: Must contain 'image' key with image source

        Returns:
            ToolResult with analysis
        """
        if self.llm_client is None:
            return ToolResult(
                success=False,
                output=None,
                error="LLM client not configured"
            )

        if not context or 'image' not in context:
            return ToolResult(
                success=False,
                output=None,
                error="No image provided in context"
            )

        # Load and encode image
        image_bytes = self._load_image(context['image'])
        if image_bytes is None:
            return ToolResult(
                success=False,
                output=None,
                error="Failed to load image"
            )

        try:
            # Call vision-capable LLM
            image_b64 = self._encode_image(image_bytes)

            response = self.llm_client.generate_with_image(
                prompt=query,
                image_base64=image_b64
            )

            return ToolResult(
                success=True,
                output=response.text,
                tokens_used=response.tokens_used
            )

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    def analyze_chart(
        self,
        image_source: Union[str, bytes, Path],
        question: Optional[str] = None
    ) -> ToolResult:
        """
        Analyze a chart or graph image.

        Args:
            image_source: Path, URL, or bytes of chart image
            question: Specific question about the chart

        Returns:
            ToolResult with chart analysis
        """
        prompt = question or "Describe this chart in detail. Include:\n1. Type of chart\n2. Axes labels and ranges\n3. Key data points\n4. Trends or patterns\n5. Main takeaways"

        return self.execute(prompt, context={'image': image_source})

    def extract_text(self, image_source: Union[str, bytes, Path]) -> ToolResult:
        """
        Extract text from an image (OCR-like functionality).

        Args:
            image_source: Path, URL, or bytes of image

        Returns:
            ToolResult with extracted text
        """
        prompt = "Extract all visible text from this image. Preserve the layout and structure as much as possible."
        return self.execute(prompt, context={'image': image_source})

    def answer_visual_question(
        self,
        image_source: Union[str, bytes, Path],
        question: str
    ) -> ToolResult:
        """
        Answer a question about an image.

        Args:
            image_source: Path, URL, or bytes of image
            question: Question to answer

        Returns:
            ToolResult with answer
        """
        return self.execute(question, context={'image': image_source})
