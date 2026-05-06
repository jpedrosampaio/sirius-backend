"""
Sirius API Backend Tests
Tests: Auth, Transactions, Credit Cards, Finance Stats, Chat
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthAndAuth:
    """Health check and authentication tests"""
    
    def test_api_root(self):
        """Test API root endpoint"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print(f"✓ API root: {data['message']}")
    
    def test_register_user(self):
        """Test user registration"""
        import uuid
        unique_email = f"test_{uuid.uuid4().hex[:8]}@test.com"
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "Test123!",
            "name": "Test User"
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_token" in data
        assert "user" in data
        assert data["user"]["email"] == unique_email
        print(f"✓ Register: {unique_email}")
        return data["session_token"]
    
    def test_login_user(self):
        """Test user login with test credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testsirius@test.com",
            "password": "Test123!"
        })
        # May fail if user doesn't exist, so we register first
        if response.status_code == 401:
            # Register the test user first
            reg_response = requests.post(f"{BASE_URL}/api/auth/register", json={
                "email": "testsirius@test.com",
                "password": "Test123!",
                "name": "Test Sirius User"
            })
            if reg_response.status_code == 200:
                response = requests.post(f"{BASE_URL}/api/auth/login", json={
                    "email": "testsirius@test.com",
                    "password": "Test123!"
                })
        
        assert response.status_code == 200
        data = response.json()
        assert "session_token" in data
        assert "user" in data
        print(f"✓ Login successful: {data['user']['email']}")
        return data["session_token"]
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "invalid@test.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        print("✓ Invalid login rejected correctly")


class TestTransactions:
    """Transaction CRUD tests"""
    
    @pytest.fixture
    def auth_token(self):
        """Get auth token for tests"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testsirius@test.com",
            "password": "Test123!"
        })
        if response.status_code == 401:
            reg_response = requests.post(f"{BASE_URL}/api/auth/register", json={
                "email": "testsirius@test.com",
                "password": "Test123!",
                "name": "Test Sirius User"
            })
            response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": "testsirius@test.com",
                "password": "Test123!"
            })
        return response.json()["session_token"]
    
    def test_create_transaction(self, auth_token):
        """Test creating a transaction"""
        response = requests.post(
            f"{BASE_URL}/api/transactions",
            json={
                "type": "expense",
                "amount": 50.0,
                "category": "alimentação",
                "description": "TEST_Almoço",
                "date": "2026-01-15"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "expense"
        assert data["amount"] == 50.0
        assert data["category"] == "alimentação"
        assert "transaction_id" in data
        print(f"✓ Transaction created: {data['transaction_id']}")
        return data["transaction_id"]
    
    def test_get_transactions(self, auth_token):
        """Test getting transactions"""
        response = requests.get(
            f"{BASE_URL}/api/transactions?month=2026-01",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Got {len(data)} transactions")


class TestCreditCards:
    """Credit card CRUD tests"""
    
    @pytest.fixture
    def auth_token(self):
        """Get auth token for tests"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testsirius@test.com",
            "password": "Test123!"
        })
        if response.status_code == 401:
            requests.post(f"{BASE_URL}/api/auth/register", json={
                "email": "testsirius@test.com",
                "password": "Test123!",
                "name": "Test Sirius User"
            })
            response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": "testsirius@test.com",
                "password": "Test123!"
            })
        return response.json()["session_token"]
    
    def test_create_credit_card(self, auth_token):
        """Test creating a credit card"""
        import uuid
        card_name = f"TEST_Card_{uuid.uuid4().hex[:4]}"
        response = requests.post(
            f"{BASE_URL}/api/credit-cards",
            json={
                "name": card_name,
                "limit": 5000.0,
                "closing_day": 15,
                "due_day": 25
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == card_name
        assert data["limit"] == 5000.0
        assert "card_id" in data
        print(f"✓ Credit card created: {data['card_id']}")
        return data["card_id"]
    
    def test_get_credit_cards(self, auth_token):
        """Test getting credit cards"""
        response = requests.get(
            f"{BASE_URL}/api/credit-cards",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Got {len(data)} credit cards")


class TestFinanceStats:
    """Finance stats endpoint tests"""
    
    @pytest.fixture
    def auth_token(self):
        """Get auth token for tests"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testsirius@test.com",
            "password": "Test123!"
        })
        if response.status_code == 401:
            requests.post(f"{BASE_URL}/api/auth/register", json={
                "email": "testsirius@test.com",
                "password": "Test123!",
                "name": "Test Sirius User"
            })
            response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": "testsirius@test.com",
                "password": "Test123!"
            })
        return response.json()["session_token"]
    
    def test_finance_stats(self, auth_token):
        """Test finance stats endpoint returns expense_by_category"""
        response = requests.get(
            f"{BASE_URL}/api/finance/stats?month=2026-01",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "expense_by_category" in data, "Missing expense_by_category field"
        assert "total_income" in data
        assert "total_expense" in data
        print(f"✓ Finance stats: income={data['total_income']}, expense={data['total_expense']}")
        print(f"✓ expense_by_category: {data['expense_by_category']}")


class TestChat:
    """Chat AI endpoint tests"""
    
    @pytest.fixture
    def auth_token(self):
        """Get auth token for tests"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testsirius@test.com",
            "password": "Test123!"
        })
        if response.status_code == 401:
            requests.post(f"{BASE_URL}/api/auth/register", json={
                "email": "testsirius@test.com",
                "password": "Test123!",
                "name": "Test Sirius User"
            })
            response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": "testsirius@test.com",
                "password": "Test123!"
            })
        return response.json()["session_token"]
    
    def test_chat_send_transaction(self, auth_token):
        """Test chat endpoint with transaction text"""
        response = requests.post(
            f"{BASE_URL}/api/chat/send",
            json={"content": "Gastei R$ 30 no café"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "user_message" in data
        assert "ai_message" in data
        assert data["user_message"]["role"] == "user"
        assert data["ai_message"]["role"] == "assistant"
        print(f"✓ Chat response received")
        print(f"  AI response: {data['ai_message']['content'][:100]}...")
    
    def test_chat_get_messages(self, auth_token):
        """Test getting chat messages"""
        response = requests.get(
            f"{BASE_URL}/api/chat/messages",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Got {len(data)} chat messages")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
