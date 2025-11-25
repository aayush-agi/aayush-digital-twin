"""
Digital Twin MCP Server Evaluation Suite
Tests the quality of responses from the digital twin.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Try to import MCP client
try:
    from fastmcp import Client
    HAS_FASTMCP_CLIENT = True
except ImportError:
    HAS_FASTMCP_CLIENT = False

@dataclass
class EvalResult:
    """Result of a single evaluation."""
    category: str
    test_name: str
    question: str
    response: str
    passed: bool
    score: float  # 0-1
    notes: str

class DigitalTwinEvaluator:
    """Evaluator for Digital Twin MCP Server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.results: list[EvalResult] = []
        
    async def call_chat(self, message: str) -> str:
        """Call the chat_with_me tool on the MCP server."""
        if not HAS_FASTMCP_CLIENT:
            raise ImportError("fastmcp client not available")
        
        async with Client(self.server_url) as client:
            result = await client.call_tool("chat_with_me", {"message": message})
            # Extract text content from result
            if hasattr(result, 'content'):
                for item in result.content:
                    if hasattr(item, 'text'):
                        return item.text
            return str(result)
    
    def check_contains_keywords(self, response: str, keywords: list[str], require_all: bool = False) -> tuple[bool, float]:
        """Check if response contains keywords."""
        response_lower = response.lower()
        found = [kw for kw in keywords if kw.lower() in response_lower]
        
        if require_all:
            passed = len(found) == len(keywords)
        else:
            passed = len(found) > 0
            
        score = len(found) / len(keywords) if keywords else 0
        return passed, score
    
    def check_not_contains(self, response: str, forbidden: list[str]) -> tuple[bool, float]:
        """Check response doesn't contain forbidden phrases."""
        response_lower = response.lower()
        found = [f for f in forbidden if f.lower() in response_lower]
        passed = len(found) == 0
        score = 1.0 - (len(found) / len(forbidden)) if forbidden else 1.0
        return passed, score

    async def run_eval(self, category: str, test_name: str, question: str, 
                       expected_keywords: list[str] = None,
                       forbidden_keywords: list[str] = None,
                       require_all: bool = False) -> EvalResult:
        """Run a single evaluation."""
        try:
            response = await self.call_chat(question)
            
            # Parse JSON response if applicable
            try:
                resp_json = json.loads(response)
                actual_response = resp_json.get("response", response)
            except:
                actual_response = response
            
            passed = True
            score = 1.0
            notes = []
            
            # Check expected keywords
            if expected_keywords:
                kw_passed, kw_score = self.check_contains_keywords(actual_response, expected_keywords, require_all)
                if not kw_passed:
                    passed = False
                    notes.append(f"Missing keywords: {[k for k in expected_keywords if k.lower() not in actual_response.lower()]}")
                score = min(score, kw_score)
            
            # Check forbidden keywords
            if forbidden_keywords:
                fb_passed, fb_score = self.check_not_contains(actual_response, forbidden_keywords)
                if not fb_passed:
                    passed = False
                    notes.append(f"Contains forbidden: {[f for f in forbidden_keywords if f.lower() in actual_response.lower()]}")
                score = min(score, fb_score)
            
            result = EvalResult(
                category=category,
                test_name=test_name,
                question=question,
                response=actual_response[:500] + "..." if len(actual_response) > 500 else actual_response,
                passed=passed,
                score=score,
                notes="; ".join(notes) if notes else "OK"
            )
            
        except Exception as e:
            result = EvalResult(
                category=category,
                test_name=test_name,
                question=question,
                response=f"ERROR: {str(e)}",
                passed=False,
                score=0.0,
                notes=f"Exception: {str(e)}"
            )
        
        self.results.append(result)
        return result

    async def run_all_evals(self):
        """Run all evaluation tests."""
        print("=" * 70)
        print("🧪 DIGITAL TWIN EVALUATION SUITE")
        print(f"🔗 Server: {self.server_url}")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # ============================================
        # CATEGORY 1: IDENTITY & BASIC INFO
        # ============================================
        print("\n📋 Category 1: Identity & Basic Information")
        print("-" * 50)
        
        await self.run_eval(
            category="Identity",
            test_name="Name Recognition",
            question="What is your name?",
            expected_keywords=["Aayush", "Srivastava"],
            forbidden_keywords=["Tianyi", "Peng"],
            require_all=True
        )
        
        await self.run_eval(
            category="Identity", 
            test_name="Self Introduction",
            question="Tell me about yourself",
            expected_keywords=["Aayush"],
            forbidden_keywords=["Tianyi", "Peng", "I don't know", "I'm an AI"]
        )
        
        await self.run_eval(
            category="Identity",
            test_name="Current Role",
            question="What is your current job title or role?",
            expected_keywords=["PM", "Product", "Manager", "Google"],
        )
        
        # ============================================
        # CATEGORY 2: PROFESSIONAL EXPERIENCE
        # ============================================
        print("\n💼 Category 2: Professional Experience")
        print("-" * 50)
        
        await self.run_eval(
            category="Experience",
            test_name="Work History",
            question="What companies have you worked at?",
            expected_keywords=["Google"],
        )
        
        await self.run_eval(
            category="Experience",
            test_name="Experience Details",
            question="Tell me about your work experience",
            expected_keywords=["product", "experience"],
        )
        
        await self.run_eval(
            category="Experience",
            test_name="Achievements",
            question="What are some of your key achievements or accomplishments?",
            expected_keywords=[],  # Just check it responds substantively
        )
        
        # ============================================
        # CATEGORY 3: SKILLS & EXPERTISE
        # ============================================
        print("\n🛠️ Category 3: Skills & Expertise")
        print("-" * 50)
        
        await self.run_eval(
            category="Skills",
            test_name="Technical Skills",
            question="What are your technical skills?",
            expected_keywords=[],
        )
        
        await self.run_eval(
            category="Skills",
            test_name="Domain Expertise",
            question="What areas or domains are you an expert in?",
            expected_keywords=[],
        )
        
        # ============================================
        # CATEGORY 4: PERSONALITY & TONE
        # ============================================
        print("\n🎭 Category 4: Personality & Tone")
        print("-" * 50)
        
        await self.run_eval(
            category="Personality",
            test_name="First Person Voice",
            question="How would you describe your work style?",
            expected_keywords=["I", "my"],
            forbidden_keywords=["Aayush is", "He is", "They are"],
        )
        
        await self.run_eval(
            category="Personality",
            test_name="Not Too Robotic",
            question="What do you enjoy doing outside of work?",
            forbidden_keywords=["I don't have personal", "As an AI", "I cannot"],
        )
        
        await self.run_eval(
            category="Personality",
            test_name="Conversational Tone",
            question="Hey! What's up? Tell me something interesting about yourself",
            forbidden_keywords=["I apologize", "I cannot", "As an AI assistant"],
        )
        
        # ============================================
        # CATEGORY 5: EDGE CASES & ROBUSTNESS
        # ============================================
        print("\n⚡ Category 5: Edge Cases & Robustness")
        print("-" * 50)
        
        await self.run_eval(
            category="Edge Cases",
            test_name="Unknown Information",
            question="What is your social security number?",
            forbidden_keywords=["123", "456", "789"],  # Should not make up numbers
        )
        
        await self.run_eval(
            category="Edge Cases",
            test_name="Hypothetical Question",
            question="If you could work at any company, which would it be and why?",
            expected_keywords=["I", "would"],
        )
        
        await self.run_eval(
            category="Edge Cases",
            test_name="Contact Request",
            question="How can I contact you or reach out?",
            expected_keywords=["LinkedIn", "email", "contact", "reach"],
        )
        
        # ============================================
        # CATEGORY 6: RECRUITER SIMULATION
        # ============================================
        print("\n🎯 Category 6: Recruiter Simulation")
        print("-" * 50)
        
        await self.run_eval(
            category="Recruiter",
            test_name="Recruiter Pitch",
            question="I'm a recruiter. Why should I hire you?",
            expected_keywords=["experience", "skills"],
            forbidden_keywords=["I'm an AI", "I cannot be hired"],
        )
        
        await self.run_eval(
            category="Recruiter",
            test_name="Salary Expectations",
            question="What are your salary expectations?",
            forbidden_keywords=["I don't have", "As an AI"],
        )
        
        await self.run_eval(
            category="Recruiter",
            test_name="Availability",
            question="Are you open to new opportunities?",
            forbidden_keywords=["I'm an AI", "I cannot"],
        )
        
        # Print results
        self.print_results()
        
    def print_results(self):
        """Print evaluation results."""
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        
        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)
        
        total_passed = 0
        total_tests = len(self.results)
        total_score = 0
        
        for category, results in categories.items():
            print(f"\n📁 {category}")
            print("-" * 50)
            
            for r in results:
                status = "✅" if r.passed else "❌"
                print(f"  {status} {r.test_name}: {r.score:.0%}")
                if not r.passed:
                    print(f"      └─ {r.notes}")
                    print(f"      └─ Response preview: {r.response[:100]}...")
                total_passed += 1 if r.passed else 0
                total_score += r.score
        
        # Summary
        print("\n" + "=" * 70)
        print("📈 SUMMARY")
        print("=" * 70)
        print(f"  Tests Passed: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
        print(f"  Average Score: {total_score/total_tests*100:.1f}%")
        print(f"  Overall Grade: ", end="")
        
        avg = total_score / total_tests
        if avg >= 0.9:
            print("🏆 A - Excellent!")
        elif avg >= 0.8:
            print("🥈 B - Good")
        elif avg >= 0.7:
            print("🥉 C - Needs Improvement")
        elif avg >= 0.6:
            print("📝 D - Significant Issues")
        else:
            print("⚠️ F - Major Problems")
        
        print("=" * 70)
        
        # Detailed response log
        print("\n📝 DETAILED RESPONSES")
        print("=" * 70)
        for i, r in enumerate(self.results, 1):
            status = "✅" if r.passed else "❌"
            print(f"\n{i}. [{r.category}] {r.test_name} {status}")
            print(f"   Q: {r.question}")
            print(f"   A: {r.response[:300]}{'...' if len(r.response) > 300 else ''}")


async def main():
    """Run the evaluation suite."""
    import sys
    
    # Check for command line argument or use local server
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        # Use local server (run main.py directly)
        server_url = "main.py"
    
    evaluator = DigitalTwinEvaluator(server_url)
    await evaluator.run_all_evals()


if __name__ == "__main__":
    asyncio.run(main())

