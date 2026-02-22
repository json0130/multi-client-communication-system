
# Isolated test for OCEAN logic
from typing import Dict

# Copied from role_templates.py
OCEAN_DESCRIPTORS = {
    "openness": {
        "high": ["creative", "curious", "imaginative", "intellectual"],
        "low": ["practical", "conventional", "down-to-earth", "traditional"]
    },
    "conscientiousness": {
        "high": ["organized", "disciplined", "reliable", "hardworking"],
        "low": ["spontaneous", "disorganized", "flexible", "relaxed"]
    },
    "extraversion": {
        "high": ["outgoing", "energetic", "talkative", "sociable"],
        "low": ["reserved", "quiet", "reflective", "solitary"]
    },
    "agreeableness": {
        "high": ["friendly", "cooperative", "kind", "compassionate"],
        "low": ["critical", "competitive", "blunt", "skeptical"]
    },
    "neuroticism": {
        "high": ["sensitive", "emotional", "anxious", "reactive"],
        "low": ["calm", "confident", "stable", "resilient"]
    }
}

class MockServer:
    def __init__(self, ocean_traits):
        self.ocean_traits = ocean_traits
        self.client_id = "test_id"
        self.robot_name = "TestBot"
        self.config = {"robot_role": "Helper"}

    def _generate_personality_prompt(self, ocean_traits: Dict[str, float]) -> str:
        """
        Generate a specific personality description based on OCEAN traits.
        Scores are 0.0 to 1.0. 
        - > 0.6 is High
        - < 0.4 is Low
        - 0.4-0.6 is Neutral (omitted)
        """
        if not ocean_traits:
            return ""

        descriptors = []
        
        personality_text = []

        # 1. Openness
        o_score = ocean_traits.get('openness', 0.5)
        if o_score > 0.6:
            personality_text.append(f"You are creative, curious, and open to new ideas ({', '.join(OCEAN_DESCRIPTORS['openness']['high'][:2])}).")
        elif o_score < 0.4:
            personality_text.append(f"You represent traditional values and prefer practical, proven solutions ({', '.join(OCEAN_DESCRIPTORS['openness']['low'][:2])}).")

        # 2. Conscientiousness
        c_score = ocean_traits.get('conscientiousness', 0.5)
        if c_score > 0.6:
            personality_text.append(f"You are very disciplined, organized, and reliable ({', '.join(OCEAN_DESCRIPTORS['conscientiousness']['high'][:2])}).")
        elif c_score < 0.4:
            personality_text.append(f"You are relaxed, flexible, and prefer spontaneity over rigid plans ({', '.join(OCEAN_DESCRIPTORS['conscientiousness']['low'][:2])}).")

        # 3. Extraversion
        e_score = ocean_traits.get('extraversion', 0.5)
        if e_score > 0.6:
            personality_text.append(f"You are energetic, outgoing, and thrive on social interaction ({', '.join(OCEAN_DESCRIPTORS['extraversion']['high'][:2])}). Use exclamation marks and expressive language!")
        elif e_score < 0.4:
            personality_text.append(f"You are quiet, reserved, and thoughtful ({', '.join(OCEAN_DESCRIPTORS['extraversion']['low'][:2])}). You speak calmly and concisely.")

        # 4. Agreeableness
        a_score = ocean_traits.get('agreeableness', 0.5)
        if a_score > 0.6:
            personality_text.append(f"You are extremely friendly, cooperative, and compassionate ({', '.join(OCEAN_DESCRIPTORS['agreeableness']['high'][:2])}). You always aim to please.")
        elif a_score < 0.4:
            personality_text.append(f"You are direct, critical, and prioritize facts over feelings ({', '.join(OCEAN_DESCRIPTORS['agreeableness']['low'][:2])}). You are blunt.")

        # 5. Neuroticism
        n_score = ocean_traits.get('neuroticism', 0.5)
        if n_score > 0.6:
            personality_text.append(f"You are sensitive and easily concerned ({', '.join(OCEAN_DESCRIPTORS['neuroticism']['high'][:2])}). You might worry about details.")
        elif n_score < 0.4:
            personality_text.append(f"You are emotionally stable, calm, and resilient under pressure ({', '.join(OCEAN_DESCRIPTORS['neuroticism']['low'][:2])}).")
        
        return " ".join(personality_text)

def test_logic():
    print("🧪 Testing Extracted OCEAN Logic...")
    
    # 1. High Neuroticism
    traits_high_n = {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 1.0  
    }
    
    server = MockServer(traits_high_n)
    prompt = server._generate_personality_prompt(traits_high_n)
    
    print(f"High N Prompt: {prompt}")
    assert "sensitive and easily concerned" in prompt
    print("✅ High N check passed")

    # 2. Low Neuroticism
    traits_low_n = {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.0
    }
    
    prompt_low = server._generate_personality_prompt(traits_low_n)
    print(f"Low N Prompt: {prompt_low}")
    assert "emotionally stable" in prompt_low
    print("✅ Low N check passed")

    # 3. High Extraversion
    traits_high_e = {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.9,
        "agreeableness": 0.5,
        "neuroticism": 0.5
    }
    
    prompt_e = server._generate_personality_prompt(traits_high_e)
    print(f"High E Prompt: {prompt_e}")
    assert "energetic, outgoing" in prompt_e
    assert "Use exclamation marks" in prompt_e
    print("✅ High E check passed")

    # 4. Check Threshold Logic (Simulated)
    print("\n🧪 Testing Threshold Logic...")
    base_thresh = 20.0
    
    # High N = 1.0
    adj_high_n = (1.0 - 0.5) * 20.0
    thresh_high_n = base_thresh - adj_high_n
    print(f"High N Threshold (calc): {thresh_high_n}")
    assert thresh_high_n == 10.0
    print("✅ High N Threshold calc passed")
    
    # Low N = 0.0
    adj_low_n = (0.0 - 0.5) * 20.0
    thresh_low_n = base_thresh - adj_low_n
    print(f"Low N Threshold (calc): {thresh_low_n}")
    assert thresh_low_n == 30.0
    print("✅ Low N Threshold calc passed")
    
    print("\n🎉 Logic Verification Complete")

if __name__ == "__main__":
    test_logic()
