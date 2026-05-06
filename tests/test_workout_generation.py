"""
Workout Generation API Tests
Tests: Two generation modes (tipo_treino, periodo), split badges on plans
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')


class TestWorkoutGeneration:
    """Workout plan generation tests for dual-mode feature"""
    
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
    
    def test_generate_periodo_mode_dia(self, auth_token):
        """Test generating workout with periodo mode (duration=dia) - fast test"""
        response = requests.post(
            f"{BASE_URL}/api/workout-plans/generate",
            json={
                "objective": "hipertrofia",
                "level": "intermediario",
                "generation_mode": "periodo",
                "duration": "dia",
                "muscle_groups": ["peito", "triceps"]
            },
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=90
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert data.get("success") == True, "Expected success=true"
        assert "xp_earned" in data, "Missing xp_earned field"
        assert "plan" in data, "Missing plan field"
        
        plan = data["plan"]
        assert plan.get("generation_mode") == "periodo", f"Expected generation_mode='periodo', got {plan.get('generation_mode')}"
        assert plan.get("split_type") is None, "split_type should be None for periodo mode"
        assert "plan_id" in plan, "Missing plan_id"
        assert "name" in plan, "Missing name"
        
        print(f"✓ Periodo mode (dia) generation: {plan['name']}")
        print(f"  Plan ID: {plan['plan_id']}")
        print(f"  XP earned: {data['xp_earned']}")
        return plan["plan_id"]
    
    def test_get_workout_plans_with_split_fields(self, auth_token):
        """Test that GET /workout-plans returns plans with split_type and generation_mode fields"""
        response = requests.get(
            f"{BASE_URL}/api/workout-plans",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        plans = response.json()
        assert isinstance(plans, list), "Expected list of plans"
        
        # Check that plans have the required fields
        found_periodo = False
        found_tipo_treino = False
        
        for plan in plans:
            # All plans should have these base fields
            assert "plan_id" in plan, "Missing plan_id"
            assert "name" in plan, "Missing name"
            
            # Check generation_mode field exists
            generation_mode = plan.get("generation_mode")
            
            if generation_mode == "periodo":
                found_periodo = True
                # Periodo mode should have null split_type
                assert plan.get("split_type") is None or plan.get("split_type") == None, \
                    f"Periodo mode should have null split_type, got {plan.get('split_type')}"
                print(f"✓ Found periodo plan: {plan['name']}")
            
            elif generation_mode == "tipo_treino":
                found_tipo_treino = True
                # Split mode should have split_type and split_config
                assert plan.get("split_type") in ["AB", "ABC", "ABCD", "ABCDE"], \
                    f"Expected valid split_type, got {plan.get('split_type')}"
                assert plan.get("split_config") is not None, "Missing split_config"
                assert isinstance(plan.get("split_config"), list), "split_config should be a list"
                
                # Verify split_config structure
                for config in plan.get("split_config", []):
                    assert "label" in config, "Missing label in split_config"
                    assert "muscle_groups" in config, "Missing muscle_groups in split_config"
                
                # Check training_days_per_week and cycle_weeks
                assert plan.get("training_days_per_week") is not None, "Missing training_days_per_week"
                assert plan.get("cycle_weeks") is not None, "Missing cycle_weeks"
                
                print(f"✓ Found tipo_treino plan: {plan['name']}")
                print(f"  Split: {plan['split_type']}, Days/week: {plan['training_days_per_week']}, Cycle: {plan['cycle_weeks']} weeks")
        
        print(f"✓ Total plans: {len(plans)}")
        print(f"  Found periodo mode: {found_periodo}")
        print(f"  Found tipo_treino mode: {found_tipo_treino}")
    
    def test_generate_periodo_mode_invalid_level(self, auth_token):
        """Test generating workout with invalid level parameter"""
        response = requests.post(
            f"{BASE_URL}/api/workout-plans/generate",
            json={
                "objective": "hipertrofia",
                "level": "invalid_level",  # Invalid
                "generation_mode": "periodo",
                "duration": "dia"
            },
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=90
        )
        # Should still work as AI will handle it, or return 422 validation error
        assert response.status_code in [200, 422], f"Unexpected status: {response.status_code}"
        print(f"✓ Invalid level handled correctly: status {response.status_code}")
    
    def test_generate_without_auth(self):
        """Test that generation requires authentication"""
        response = requests.post(
            f"{BASE_URL}/api/workout-plans/generate",
            json={
                "objective": "hipertrofia",
                "level": "intermediario",
                "generation_mode": "periodo",
                "duration": "dia"
            },
            timeout=10
        )
        assert response.status_code == 401, f"Expected 401 Unauthorized, got {response.status_code}"
        print("✓ Unauthenticated request correctly rejected")


class TestWorkoutPlansAPI:
    """Basic workout plans API tests"""
    
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
    
    def test_get_workout_plans(self, auth_token):
        """Test getting workout plans"""
        response = requests.get(
            f"{BASE_URL}/api/workout-plans",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ Got {len(data)} workout plans")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
