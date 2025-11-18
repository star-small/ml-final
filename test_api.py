"""
API Test Script - Student Mental Health Classifier
Tests all endpoints and various prediction scenarios
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 10

class APITester:
    """Test suite for the Student Mental Health Classifier API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print("\\n[1/7] Testing Health Check...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            assert data['status'] == 'healthy', "Status not healthy"
            assert data['model_loaded'] == True, "Model not loaded"
            
            print("✅ Health check passed")
            print(f"   Status: {data['status']}, Model: {data['model_loaded']}")
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        print("\\n[2/7] Testing Model Info...")
        try:
            response = self.session.get(f"{self.base_url}/model-info", timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            assert data['model_name'] == 'XGBoost', "Wrong model"
            assert len(data['classes']) == 3, "Wrong number of classes"
            
            print("✅ Model info passed")
            print(f"   Model: {data['model_name']}, Classes: {data['classes']}")
            return True
        except Exception as e:
            print(f"❌ Model info test failed: {e}")
            return False
    
    def test_single_prediction_low_stress(self) -> bool:
        """Test prediction for LOW stress student"""
        print("\\n[3/7] Testing Prediction (LOW STRESS)...")
        
        payload = {
            "sleep_hours": 8,
            "study_hours_per_day": 3,
            "social_interaction_score": 9,
            "exercise_hours_per_week": 6,
            "academic_performance": 90,
            "exam_anxiety_level": 2,
            "family_income_level": "medium",
            "caffeine_intake": 1,
            "assignment_overload": 2,
            "extracurricular_activities": 3
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            assert data['stress_level'] in ['Low', 'Medium', 'High'], "Invalid stress level"
            assert 0 <= data['confidence'] <= 1, "Invalid confidence"
            assert len(data['recommendations']) > 0, "No recommendations"
            
            print(f"✅ Prediction passed")
            print(f"   Stress Level: {data['stress_level']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            print(f"   Probabilities: {json.dumps(data['probabilities'], indent=4)}")
            
            return True
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            return False
    
    def test_single_prediction_high_stress(self) -> bool:
        """Test prediction for HIGH stress student"""
        print("\\n[4/7] Testing Prediction (HIGH STRESS)...")
        
        payload = {
            "sleep_hours": 4,
            "study_hours_per_day": 8,
            "social_interaction_score": 3,
            "exercise_hours_per_week": 0,
            "academic_performance": 60,
            "exam_anxiety_level": 9,
            "family_income_level": "low",
            "caffeine_intake": 4,
            "assignment_overload": 9,
            "extracurricular_activities": 0
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ High stress prediction passed")
            print(f"   Stress Level: {data['stress_level']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            print(f"   Recommendations provided: {len(data['recommendations'])} items")
            
            return True
        except Exception as e:
            print(f"❌ High stress prediction test failed: {e}")
            return False
    
    def test_single_prediction_medium_stress(self) -> bool:
        """Test prediction for MEDIUM stress student"""
        print("\\n[5/7] Testing Prediction (MEDIUM STRESS)...")
        
        payload = {
            "sleep_hours": 6.5,
            "study_hours_per_day": 5,
            "social_interaction_score": 6,
            "exercise_hours_per_week": 3,
            "academic_performance": 75,
            "exam_anxiety_level": 6,
            "family_income_level": "medium",
            "caffeine_intake": 2.5,
            "assignment_overload": 6,
            "extracurricular_activities": 2
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ Medium stress prediction passed")
            print(f"   Stress Level: {data['stress_level']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            
            return True
        except Exception as e:
            print(f"❌ Medium stress prediction test failed: {e}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        print("\\n[6/7] Testing Batch Prediction...")
        
        payloads = [
            {
                "sleep_hours": 7,
                "study_hours_per_day": 4,
                "social_interaction_score": 8,
                "exercise_hours_per_week": 5,
                "academic_performance": 85,
                "exam_anxiety_level": 5,
                "family_income_level": "medium",
                "caffeine_intake": 2,
                "assignment_overload": 6,
                "extracurricular_activities": 2
            },
            {
                "sleep_hours": 5,
                "study_hours_per_day": 7,
                "social_interaction_score": 4,
                "exercise_hours_per_week": 1,
                "academic_performance": 65,
                "exam_anxiety_level": 8,
                "family_income_level": "low",
                "caffeine_intake": 3,
                "assignment_overload": 8,
                "extracurricular_activities": 0
            }
        ]
        
        try:
            response = self.session.post(
                f"{self.base_url}/batch-predict",
                json=payloads,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            assert data['count'] == len(payloads), "Not all predictions processed"
            
            print(f"✅ Batch prediction passed")
            print(f"   Processed: {data['count']} students")
            for i, pred in enumerate(data['predictions'], 1):
                print(f"   Student {i}: {pred['stress_level']} (confidence: {pred['confidence']:.2%})")
            
            return True
        except Exception as e:
            print(f"❌ Batch prediction test failed: {e}")
            return False
    
    def test_input_validation(self) -> bool:
        """Test input validation"""
        print("\\n[7/7] Testing Input Validation...")
        
        # Invalid family_income_level
        invalid_payload = {
            "sleep_hours": 7,
            "study_hours_per_day": 4,
            "social_interaction_score": 8,
            "exercise_hours_per_week": 5,
            "academic_performance": 85,
            "exam_anxiety_level": 5,
            "family_income_level": "invalid",  # ❌ Invalid
            "caffeine_intake": 2,
            "assignment_overload": 6,
            "extracurricular_activities": 2
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_payload,
                timeout=TIMEOUT
            )
            
            # Should fail with 422 status code
            if response.status_code == 422:
                print("✅ Input validation test passed")
                print(f"   Correctly rejected invalid input (status: {response.status_code})")
                return True
            else:
                print(f"❌ Validation error not caught (status: {response.status_code})")
                return False
        
        except Exception as e:
            print(f"❌ Input validation test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return summary"""
        print("=" * 60)
        print("🧠 STUDENT MENTAL HEALTH CLASSIFIER - API TEST SUITE")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        
        start_time = time.time()
        
        results = {
            "health_check": self.test_health_check(),
            "model_info": self.test_model_info(),
            "prediction_low": self.test_single_prediction_low_stress(),
            "prediction_high": self.test_single_prediction_high_stress(),
            "prediction_medium": self.test_single_prediction_medium_stress(),
            "batch_prediction": self.test_batch_prediction(),
            "input_validation": self.test_input_validation(),
        }
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print("\\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"Passed: {passed}/{total}")
        print(f"Time: {elapsed_time:.2f}s")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {test_name}")
        
        if passed == total:
            print("\\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\\n⚠️  {total - passed} test(s) failed")
        
        print("=" * 60)
        
        return {
            "passed": passed,
            "total": total,
            "time_elapsed": elapsed_time,
            "results": results
        }

def main():
    """Main test function"""
    print("\\nAttempting to connect to API...")
    
    tester = APITester()
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        response.raise_for_status()
        print("✅ API is running and responding\\n")
    except requests.exceptions.ConnectionError:
        print(f"\\n❌ ERROR: Cannot connect to API at {BASE_URL}")
        print("Make sure to start the API first:")
        print("  python app.py")
        print("\\nAlternatively, if using a different URL:")
        print("  python test_api.py <url>")
        exit(1)
    except Exception as e:
        print(f"\\n❌ ERROR: {e}")
        exit(1)
    
    # Run tests
    summary = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if summary['passed'] == summary['total'] else 1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    
    main()
