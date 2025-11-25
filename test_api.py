#!/usr/bin/env python3
"""
Test script for RAG Repository Analyzer API
"""

import requests
import json
import time
import sys
from typing import Dict, Any


class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        print("🏥 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_status_endpoint(self) -> bool:
        """Test the status endpoint"""
        print("📊 Testing status endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status check passed: {data}")
                return True
            else:
                print(f"❌ Status check failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Status check error: {str(e)}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root endpoint"""
        print("🏠 Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint passed: API name = {data.get('name')}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {str(e)}")
            return False
    
    def test_analyze_endpoint(self, repository: str) -> bool:
        """Test the analyze endpoint"""
        print(f"🔍 Testing analyze endpoint with repository: {repository}")
        try:
            payload = {"repository": repository}
            response = self.session.post(
                f"{self.base_url}/analyze",
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Analyze passed: {data.get('message')}")
                return True
            else:
                print(f"❌ Analyze failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Analyze error: {str(e)}")
            return False
    
    def test_query_endpoint(self, question: str) -> bool:
        """Test the query endpoint"""
        print(f"❓ Testing query endpoint with question: '{question}'")
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/query",
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Query passed!")
                print(f"   Answer: {data.get('answer', 'No answer')[:100]}...")
                print(f"   Sources: {len(data.get('sources', []))} chunks found")
                return True
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return False
    
    def test_invalid_requests(self) -> bool:
        """Test error handling with invalid requests"""
        print("🚫 Testing error handling...")
        
        tests_passed = 0
        total_tests = 4
        
        # Test invalid repository format
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                data=json.dumps({"repository": "invalid-format"})
            )
            if response.status_code == 400:
                print("✅ Invalid repository format correctly rejected")
                tests_passed += 1
            else:
                print(f"❌ Invalid repository should return 400, got {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing invalid repository: {str(e)}")
        
        # Test empty question
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                data=json.dumps({"question": ""})
            )
            if response.status_code == 400:
                print("✅ Empty question correctly rejected")
                tests_passed += 1
            else:
                print(f"❌ Empty question should return 400, got {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing empty question: {str(e)}")
        
        # Test query without analysis
        try:
            # First reset by calling a different repository
            self.session.post(
                f"{self.base_url}/analyze",
                data=json.dumps({"repository": "nonexistent/repo"})
            )
            # Then try to query
            response = self.session.post(
                f"{self.base_url}/query",
                data=json.dumps({"question": "test question"})
            )
            if response.status_code in [400, 500]:
                print("✅ Query without analysis correctly handled")
                tests_passed += 1
            else:
                print(f"❌ Query without analysis should return error, got {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing query without analysis: {str(e)}")
        
        # Test malformed JSON
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                data="invalid json"
            )
            if response.status_code == 422:  # FastAPI returns 422 for validation errors
                print("✅ Malformed JSON correctly rejected")
                tests_passed += 1
            else:
                print(f"❌ Malformed JSON should return 422, got {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing malformed JSON: {str(e)}")
        
        return tests_passed == total_tests
    
    def run_full_test_suite(self, repository: str = "microsoft/vscode", 
                           question: str = "What is this repository about?") -> bool:
        """Run the complete test suite"""
        print("🚀 Starting RAG Repository Analyzer API Test Suite")
        print("=" * 60)
        
        test_results = []
        
        # Test basic endpoints
        test_results.append(("Health Check", self.test_health_endpoint()))
        test_results.append(("Status Check", self.test_status_endpoint()))
        test_results.append(("Root Endpoint", self.test_root_endpoint()))
        
        # Test error handling
        test_results.append(("Error Handling", self.test_invalid_requests()))
        
        # Test main workflow (this will take time)
        print(f"\n⏳ Starting repository analysis (this may take several minutes)...")
        test_results.append(("Repository Analysis", self.test_analyze_endpoint(repository)))
        
        # If analysis succeeded, test query
        if test_results[-1][1]:  # If analyze passed
            test_results.append(("Repository Query", self.test_query_endpoint(question)))
        else:
            print("⚠️  Skipping query test due to analysis failure")
            test_results.append(("Repository Query", False))
        
        # Print results
        print("\n" + "=" * 60)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed_tests = 0
        for test_name, passed in test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:.<30} {status}")
            if passed:
                passed_tests += 1
        
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")
            return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG Repository Analyzer API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--repository", default="microsoft/vscode",
                       help="Repository to test with (default: microsoft/vscode)")
    parser.add_argument("--question", default="What is this repository about?",
                       help="Question to test with")
    parser.add_argument("--quick", action="store_true",
                       help="Run only quick tests (skip repository analysis)")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.quick:
        print("🏃 Running quick tests only...")
        success = True
        success &= tester.test_health_endpoint()
        success &= tester.test_status_endpoint()
        success &= tester.test_root_endpoint()
        success &= tester.test_invalid_requests()
        
        if success:
            print("✅ Quick tests passed!")
        else:
            print("❌ Some quick tests failed!")
            sys.exit(1)
    else:
        success = tester.run_full_test_suite(args.repository, args.question)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()