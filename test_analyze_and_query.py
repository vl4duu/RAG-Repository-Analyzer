#!/usr/bin/env python3
"""
Test script for the new /analyze-and-query endpoint
"""

import sys
import time

import requests


def test_analyze_and_query_endpoint(base_url: str = "http://localhost:8000") -> bool:
    """
    Test the new /analyze-and-query endpoint
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if test passes, False otherwise
    """
    print("🧪 Testing /analyze-and-query endpoint...")

    # Test data
    test_request = {
        "repository": "vl4duu/locomobile.co",  # Small test repository
        "question": "What is this repository about?"
    }

    try:
        # Test the new combined endpoint
        print(f"📡 Sending request to {base_url}/analyze-and-query")
        print(f"📋 Repository: {test_request['repository']}")
        print(f"❓ Question: {test_request['question']}")

        start_time = time.time()

        response = requests.post(
            f"{base_url}/analyze-and-query",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout for analysis
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"⏱️  Request completed in {duration:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
        # Assert successful response
        assert response.status_code == 200, f"Request failed with status code: {response.status_code}, Error: {response.text}"

        result = response.json()
        print("✅ Success! Response received:")

        # Assert required response fields exist
        assert 'status' in result, "Response missing 'status' field"
        assert 'repository' in result, "Response missing 'repository' field"
        assert 'answer' in result, "Response missing 'answer' field"
        assert 'sources' in result, "Response missing 'sources' field"

        print(f"   Status: {result.get('status')}")
        print(f"   Repository: {result.get('repository')}")
        print(f"   Answer: {result.get('answer', '')[:100]}...")
        sources = result.get('sources', [])
        print(f"   Sources found: {len(sources)}")

        # Validate source structure for frontend compatibility (file metadata)
        if sources:
            print("🔍 Validating file metadata in sources...")
            required_fields = {'fileName', 'file_contents', 'file_name', 'content', 'content_type', 'score'}

            for i, source in enumerate(sources):
                source_fields = set(source.keys())
                missing_fields = required_fields - source_fields

                # Assert all required fields are present
                assert not missing_fields, f"Source {i + 1} missing required fields: {missing_fields}"

                # Assert specific new fields for frontend are not empty
                assert source.get('fileName'), f"Source {i + 1} has empty or missing fileName field"
                assert source.get('file_contents'), f"Source {i + 1} has empty or missing file_contents field"

                # Show first source details for verification
                if i == 0:
                    print(f"   Sample source fields: {list(source.keys())}")
                    print(f"   fileName: {source.get('fileName', 'N/A')}")
                    print(f"   file_contents length: {len(source.get('file_contents', ''))}")
                    print(f"   file_name: {source.get('file_name', 'N/A')}")
                    print(f"   content_type: {source.get('content_type', 'N/A')}")
                    print(f"   score: {source.get('score', 'N/A')}")

            print("✅ All sources have required file metadata fields for frontend compatibility")
        else:
            print("⚠️  No sources returned - cannot validate file metadata structure")

        print(f"   Message: {result.get('message')}")
        return True

    except requests.exceptions.Timeout:
        print("❌ Request timed out. This is normal for large repositories.")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_validation_errors(base_url: str = "http://localhost:8000") -> bool:
    """Test validation errors for the endpoint"""
    print("\n🧪 Testing validation errors...")

    test_cases = [
        {"repository": "", "question": "What is this?"},  # Empty repository
        {"repository": "test/repo", "question": ""},  # Empty question
        {"repository": "", "question": ""},  # Both empty
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"  Test {i}: {test_case}")

        try:
            response = requests.post(
                f"{base_url}/analyze-and-query",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            # Assert that validation errors return 400 Bad Request
            assert response.status_code == 400, f"Expected 400 Bad Request for invalid input {test_case}, got {response.status_code}"
            print(f"  ✅ Correctly returned 400 Bad Request")

        except requests.exceptions.RequestException as req_e:
            raise AssertionError(f"Request error testing validation case {test_case}: {str(req_e)}")
        except Exception as e:
            raise AssertionError(f"Unexpected error testing validation case {test_case}: {str(e)}")

    return True


def main():
    """Main test function"""
    print("🚀 Testing the new /analyze-and-query endpoint\n")

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    # Test health endpoint first
    try:
        print(f"🏥 Testing health endpoint at {base_url}/health")
        health_response = requests.get(f"{base_url}/health", timeout=5)

        # Assert health endpoint returns 200
        assert health_response.status_code == 200, f"API health check failed: {health_response.status_code}"
        print("✅ API is healthy")

    except requests.exceptions.RequestException as req_e:
        raise AssertionError(
            f"Could not reach API: {str(req_e)}. Make sure the API server is running with: python -m uvicorn src.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        raise AssertionError(f"Unexpected error during health check: {str(e)}")

    # Test validation errors (quick tests)
    if not test_validation_errors(base_url):
        print("\n❌ Validation tests failed")
        return False

    print("\n" + "=" * 60)
    print("⚠️  WARNING: The next test will analyze a real GitHub repository.")
    print("   This may take several minutes and requires valid API keys.")
    print("   Press Ctrl+C to skip this test.")
    print("=" * 60)

    try:
        input("\nPress Enter to continue or Ctrl+C to skip...")
    except KeyboardInterrupt:
        print("\n⏭️  Skipping repository analysis test")
        print("✅ Validation tests passed!")
        return True

    # Test the actual endpoint
    if test_analyze_and_query_endpoint(base_url):
        print("\n🎉 All tests passed! The /analyze-and-query endpoint is working correctly.")
        return True
    else:
        print("\n❌ Endpoint test failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
