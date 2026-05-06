"""
Test New Features - Iteration 3:
1. Health condition field in workout generation
2. Import meal plan endpoint
3. Enhanced chat intent classification
4. Context-aware AI assistant
"""

import pytest
import requests
import os
import io

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
if not BASE_URL:
    raise ValueError("REACT_APP_BACKEND_URL environment variable not set")

# Test credentials
TEST_EMAIL = "testsirius@test.com"
TEST_PASSWORD = "Test123!"


class TestAuthentication:
    """Test login and get session"""
    
    @pytest.fixture(scope="class")
    def session(self):
        """Create a requests session for tests"""
        return requests.Session()
    
    def test_login_success(self, session):
        """Test login with test credentials"""
        response = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert response.status_code == 200, f"Login failed: {response.text}"
        data = response.json()
        assert "user" in data or "user_id" in data, "No user data in response"
        print(f"✅ Login successful for {TEST_EMAIL}")


class TestChatIntentClassification:
    """Test the enhanced intent classification for chat"""
    
    @pytest.fixture(scope="class")
    def auth_session(self):
        """Get authenticated session"""
        session = requests.Session()
        login_resp = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        return session

    def test_general_question_returns_general_intent(self, auth_session):
        """Test 'como estão minhas metas?' returns intent='general' (not 'goal')"""
        response = auth_session.post(f"{BASE_URL}/api/chat/general", json={
            "content": "como estão minhas metas?"
        })
        # May get 200 or 429 (rate limit)
        if response.status_code == 429:
            pytest.skip("Gemini API rate limited (429) - expected behavior during testing")
        assert response.status_code == 200, f"Chat failed: {response.status_code} - {response.text}"
        data = response.json()
        intent = data.get("intent", "")
        # 'como estão minhas metas?' should be general (asking about status), not goal (creating)
        assert intent == "general", f"Expected intent='general', got '{intent}'"
        print(f"✅ 'como estão minhas metas?' correctly returns intent='general'")

    def test_create_goal_returns_goal_intent(self, auth_session):
        """Test 'criar meta: economizar 5000' returns intent='goal'"""
        response = auth_session.post(f"{BASE_URL}/api/chat/general", json={
            "content": "criar meta: economizar 5000"
        })
        # May get 200 or 429 (rate limit)
        if response.status_code == 429:
            pytest.skip("Gemini API rate limited (429) - expected behavior during testing")
        assert response.status_code == 200, f"Chat failed: {response.status_code} - {response.text}"
        data = response.json()
        intent = data.get("intent", "")
        assert intent == "goal", f"Expected intent='goal', got '{intent}'"
        print(f"✅ 'criar meta: economizar 5000' correctly returns intent='goal'")


class TestWorkoutGenerationWithHealthCondition:
    """Test workout generation with health_condition field"""
    
    @pytest.fixture(scope="class")
    def auth_session(self):
        """Get authenticated session"""
        session = requests.Session()
        login_resp = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        return session

    def test_workout_generate_accepts_health_condition(self, auth_session):
        """Test POST /api/workout-plans/generate accepts health_condition field"""
        payload = {
            "objective": "hipertrofia",
            "level": "intermediario",
            "duration": "dia",
            "generation_mode": "periodo",
            "health_condition": "Dor no joelho direito ao agachar"
        }
        response = auth_session.post(f"{BASE_URL}/api/workout-plans/generate", json=payload)
        # May get 200, 201, 429 (rate limit), or 500 (with quota error from Gemini)
        if response.status_code == 429:
            pytest.skip("Gemini API rate limited (429) - expected behavior during testing")
        if response.status_code == 500:
            # Check if it's a quota-related error (expected during testing)
            if "429" in response.text or "RESOURCE_EXHAUSTED" in response.text or "quota" in response.text.lower():
                pytest.skip("Gemini API quota exhausted (500 with 429 detail) - expected during testing")
        # Accept 200 or 201 as success
        assert response.status_code in [200, 201, 503], f"Workout generation failed: {response.status_code} - {response.text}"
        if response.status_code == 503:
            pytest.skip("AI service unavailable (503)")
        print(f"✅ POST /api/workout-plans/generate accepts health_condition field")


class TestNutritionImportPlan:
    """Test the import meal plan endpoint"""
    
    @pytest.fixture(scope="class")
    def auth_session(self):
        """Get authenticated session"""
        session = requests.Session()
        login_resp = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        return session

    def test_import_plan_endpoint_exists(self, auth_session):
        """Test that /api/nutrition/import-plan endpoint exists and rejects invalid requests"""
        # Send empty request - should get 422 (validation error) not 404
        response = auth_session.post(f"{BASE_URL}/api/nutrition/import-plan")
        # Should return 422 (validation error for missing file) not 404
        assert response.status_code != 404, "Endpoint /api/nutrition/import-plan not found"
        assert response.status_code == 422, f"Expected 422 for missing file, got {response.status_code}"
        print(f"✅ POST /api/nutrition/import-plan endpoint exists")

    def test_import_plan_rejects_invalid_file_type(self, auth_session):
        """Test that import rejects non-allowed file types"""
        # Create a fake text file
        files = {
            'file': ('test.txt', io.BytesIO(b'This is not a valid meal plan'), 'text/plain')
        }
        response = auth_session.post(f"{BASE_URL}/api/nutrition/import-plan", files=files)
        # Should reject with 400 (invalid format)
        assert response.status_code == 400, f"Expected 400 for invalid file type, got {response.status_code}"
        assert "Formato não suportado" in response.text or "format" in response.text.lower()
        print(f"✅ Import endpoint correctly rejects invalid file types")

    def test_import_plan_accepts_valid_file_type(self, auth_session):
        """Test that import accepts valid file types (PNG)"""
        # Create a minimal PNG file header (valid PNG but minimal content)
        # PNG signature bytes
        png_header = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        # IHDR chunk (minimal 1x1 image)
        ihdr = bytes([
            0x00, 0x00, 0x00, 0x0D,  # Length: 13
            0x49, 0x48, 0x44, 0x52,  # IHDR
            0x00, 0x00, 0x00, 0x01,  # Width: 1
            0x00, 0x00, 0x00, 0x01,  # Height: 1
            0x08, 0x02,              # Bit depth: 8, Color type: 2 (RGB)
            0x00, 0x00, 0x00,        # Compression, filter, interlace
            0x90, 0x77, 0x53, 0xDE   # CRC
        ])
        # IDAT chunk (minimal compressed data)
        idat = bytes([
            0x00, 0x00, 0x00, 0x0C,  # Length
            0x49, 0x44, 0x41, 0x54,  # IDAT
            0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0xFF, 0x00, 0x05, 0xFE, 0x02, 0xFE,
            0xA3, 0x6B, 0x36, 0x01   # CRC
        ])
        # IEND chunk
        iend = bytes([
            0x00, 0x00, 0x00, 0x00,  # Length: 0
            0x49, 0x45, 0x4E, 0x44,  # IEND
            0xAE, 0x42, 0x60, 0x82   # CRC
        ])
        
        png_data = png_header + ihdr + idat + iend
        
        files = {
            'file': ('meal_plan.png', io.BytesIO(png_data), 'image/png')
        }
        response = auth_session.post(f"{BASE_URL}/api/nutrition/import-plan", files=files)
        # May get 200 (success), 429 (rate limit), or 503 (AI unavailable), or 500 (Gemini error)
        if response.status_code == 429:
            pytest.skip("Gemini API rate limited (429)")
        if response.status_code == 503:
            pytest.skip("AI service unavailable (503)")
        if response.status_code == 500:
            # Check if it's a Gemini-related error (expected during testing)
            if "gemini" in response.text.lower() or "api" in response.text.lower() or "generation" in response.text.lower():
                pytest.skip("Gemini API error (500) - expected during testing with minimal test image")
        # Accept that the file type was accepted (even if AI processing fails)
        assert response.status_code != 400, f"Valid PNG file type was rejected: {response.text}"
        print(f"✅ Import endpoint accepts PNG file type (status: {response.status_code})")


class TestChatContextEnrichment:
    """Test that general chat has full user context"""
    
    @pytest.fixture(scope="class")
    def auth_session(self):
        """Get authenticated session"""
        session = requests.Session()
        login_resp = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        return session

    def test_chat_responds_with_context(self, auth_session):
        """Test that chat responds to general questions with user context"""
        response = auth_session.post(f"{BASE_URL}/api/chat/general", json={
            "content": "resuma meu progresso geral"
        })
        if response.status_code == 429:
            pytest.skip("Gemini API rate limited (429)")
        assert response.status_code == 200, f"Chat failed: {response.status_code} - {response.text}"
        data = response.json()
        # Chat may return 'response' or 'ai_message' depending on implementation
        has_response = "response" in data or "ai_message" in data
        assert has_response, f"No response in chat reply. Keys: {data.keys()}"
        
        # Check for Gemini quota error in ai_message
        if "ai_message" in data:
            ai_content = data["ai_message"].get("content", "")
            if "429" in ai_content or "RESOURCE_EXHAUSTED" in ai_content or "quota" in ai_content.lower():
                pytest.skip("Gemini API quota exhausted - expected during testing")
        
        # The response should have the correct structure
        assert "intent" in data, "No intent in chat response"
        assert data["intent"] == "general", f"Expected intent='general', got '{data['intent']}'"
        print(f"✅ Chat responds with context-aware reply structure")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
