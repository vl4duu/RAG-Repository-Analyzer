import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Mock OpenAI before importing modules that use it
mock_openai = MagicMock()
mock_openai_module = MagicMock()
mock_openai_module.OpenAI = MagicMock(return_value=mock_openai)

with patch.dict('sys.modules', {'openai': mock_openai_module}):
    from src.embedding import embed_textual_metadata
    from src.rag_service import RAGService

class TestOpenAIIntegration(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_openai.reset_mock()
        mock_openai.embeddings.create.reset_mock()
        mock_openai.chat.completions.create.reset_mock()

    @patch('src.embedding._openai_client', mock_openai)
    @patch('src.embedding.OPENAI_API_KEY', 'test-key')
    def test_embed_textual_metadata_v1(self):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.embeddings.create.return_value = mock_response
        
        result = embed_textual_metadata("test content")
        
        # Verify call to v1.0+ API
        mock_openai.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="test content"
        )
        self.assertEqual(result, [0.1, 0.2, 0.3])

    @patch('src.rag_service._openai_client', mock_openai)
    def test_rag_service_query_ai_model_v1(self):
        # Setup mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "mocked answer"
        mock_response.choices = [mock_choice]
        mock_openai.chat.completions.create.return_value = mock_response
        
        service = RAGService()
        import asyncio
        loop = asyncio.get_event_loop()
        answer = loop.run_until_complete(service._query_ai_model("test prompt"))
        
        # Verify call to v1.0+ API
        mock_openai.chat.completions.create.assert_called_once()
        args, kwargs = mock_openai.chat.completions.create.call_args
        self.assertEqual(kwargs['model'], "gpt-3.5-turbo")
        self.assertEqual(kwargs['messages'][1]['content'], "test prompt")
        self.assertEqual(answer, "mocked answer")

if __name__ == '__main__':
    unittest.main()
