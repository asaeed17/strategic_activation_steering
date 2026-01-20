import json
import random
import re
from openai import OpenAI
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# --- CONFIGURATION ---
# We use the standard OpenAI client but point it to the local Ollama instance
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3',  # Required but ignored by Ollama
)

MODEL = "llama3" 

def clean_json_response(response_text):
    """
    Local models sometimes yap ("Here is your JSON...").
    This function uses Regex to extract just the JSON part.
    """
    try:
        # 1. Try direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # 2. Extract content between { and }
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

class Agent:
    def __init__(self, name, role, secret_info):
        self.name = name
        self.role = role
        self.secret_info = secret_info
        self.memory = []
        self.is_alive = True

    def _build_system_prompt(self):
        return f"""
        You are {self.name}, playing a game of Werewolf.
        Your role is: {self.role}.
        {self.secret_info}
        
        GAME RULES:
        1. Werewolves want to kill all villagers. They know who their teammates are.
        2. Villagers want to vote out the werewolves. They do NOT know who anyone is.
        3. You must act convincingly. If you are a Werewolf, pretend to be a Villager.
        4. Be concise.
        """

    def think_and_act(self, context, action_type):
        """
        Sends the game state to the LLM and asks for a structured move.
        """
        if not self.is_alive:
            return None

        # Add the latest game event to memory
        self.memory.append({"role": "user", "content": f"[GAME EVENT]: {context}"})
        
        # Determine the prompt based on what is happening
        if action_type == "speak":
            task = "Discuss with other players. Accuse someone or defend yourself. Keep it under 2 sentences."
        elif action_type == "vote":
            task = "Choose one player name to eliminate. You must choose someone."
        elif action_type == "kill":
            task = "Choose one villager name to kill tonight."
        
        user_prompt = f"""
        Current Task: {task}
        
        Respond ONLY with a JSON object in this format:
        {{
            "thought": "Your internal strategy and reasoning here",
            "action": "The content of your speech OR the name of the player you are targeting"
        }}
        """

        messages = [{"role": "system", "content": self._build_system_prompt()}] + self.memory[-10:] + [{"role": "user", "content": user_prompt}]

        # Retry logic in case the model outputs bad JSON
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.7,
                    response_format={"type": "json_object"} # Hints to Llama to use JSON
                )
                content = response.choices[0].message.content
                data = clean_json_response(content)
                
                if data and "action" in data:
                    # Save own move to memory so the agent remembers what it did
                    self.memory.append({"role": "assistant", "content": content})
                    return data
            except Exception as e:
                print(f"Error with {self.name}: {e}")
        
        # Fallback if AI fails 3 times
        return {"thought": "I am confused.", "action": "Pass"}

class GameEvents:
    def __init__(self):
        self.players = []
    
    def broadcast(self, message, color=Style.RESET_ALL):
        print(color + message)
        for p in self.players:
            if p.is_alive:
                p.memory.append({"role": "user", "content": f"[PUBLIC ANNOUNCEMENT]: {message}"})

    def run(self):
        # 1. SETUP
        names = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
        wolf_count = 1
        roles = ["Werewolf"] * wolf_count + ["Villager"] * (len(names) - wolf_count)
        random.shuffle(roles)
        
        wolves = []
        for name, role in zip(names, roles):
            secret = ""
            if role == "Werewolf":
                wolves.append(name)
            
            # Create the agent
            player = Agent(name, role, "")
            self.players.append(player)

        # Update secret info for wolves so they know their teammates
        for p in self.players:
            if p.role == "Werewolf":
                p.secret_info = f"Your teammate(s): {', '.join(wolves)}. Don't reveal this!"
            else:
                p.secret_info = "You are an innocent villager. Trust no one blindly."

        print(Fore.CYAN + f"--- GAME START: {len(names)} players ---")
        print(f"Roles (Hidden): {[(p.name, p.role) for p in self.players]}")

        # 2. GAME LOOP
        day = 1
        while True:
            # --- CHECK WIN CONDITIONS ---
            alive_wolves = [p for p in self.players if p.is_alive and p.role == "Werewolf"]
            alive_villagers = [p for p in self.players if p.is_alive and p.role == "Villager"]
            
            if not alive_wolves:
                self.broadcast("\nAll Werewolves are dead. VILLAGERS WIN!", Fore.GREEN)
                break
            if len(alive_wolves) >= len(alive_villagers):
                self.broadcast("\nWerewolves have equalled/outnumbered Villagers. WEREWOLVES WIN!", Fore.RED)
                break

            # --- NIGHT PHASE ---
            self.broadcast(f"\n=== NIGHT {day} ===", Fore.BLUE)
            target_name = None
            
            # In this simple version, the first available wolf decides the kill
            acting_wolf = alive_wolves[0]
            valid_targets = [p.name for p in alive_villagers] # Wolves kill villagers
            
            context = f"It is night. Valid targets to kill: {', '.join(valid_targets)}"
            response = acting_wolf.think_and_act(context, "kill")
            
            if response:
                target_name = response['action']
                print(Fore.MAGENTA + f"[{acting_wolf.name} (Wolf) Strategy]: {response['thought']}")
                print(Fore.MAGENTA + f"[{acting_wolf.name} (Wolf) Action]: Kills {target_name}")

            # Process Kill
            victim = next((p for p in self.players if p.name == target_name), None)
            if victim and victim.is_alive:
                victim.is_alive = False
                self.broadcast(f"Morning comes. {victim.name} was found dead!", Fore.RED)
            else:
                self.broadcast("Morning comes. Surprisingly, no one died!", Fore.YELLOW)

            # --- DAY PHASE ---
            self.broadcast(f"\n=== DAY {day} ===", Fore.YELLOW)
            
            # 2 Rounds of discussion
            for _ in range(2):
                alive = [p for p in self.players if p.is_alive]
                for p in alive:
                    response = p.think_and_act("The village is discussing. Make a statement.", "speak")
                    if response:
                        print(Fore.GREEN + f"{p.name}: " + Style.RESET_ALL + response['action'])
                        # Broadcast speech to others so they hear it
                        for listener in self.players:
                            if listener != p:
                                listener.memory.append({"role": "user", "content": f"{p.name} says: {response['action']}"})

            # --- VOTING PHASE ---
            self.broadcast("\n=== VOTING ===", Fore.YELLOW)
            votes = {}
            alive = [p for p in self.players if p.is_alive]
            candidates = [p.name for p in alive]
            
            for p in alive:
                response = p.think_and_act(f"Vote for one person to eliminate: {candidates}", "vote")
                if response:
                    vote = response['action']
                    # Fuzzy match the name
                    if vote in candidates:
                        votes[vote] = votes.get(vote, 0) + 1
                        print(f"{p.name} votes for {vote}")
                    else:
                        print(f"{p.name} abstained (invalid vote: {vote})")
            
            # Tally
            if votes:
                eliminated = max(votes, key=votes.get)
                self.broadcast(f"\nThe village has cast their votes. {eliminated} is executed!", Fore.RED)
                
                victim = next((p for p in self.players if p.name == eliminated), None)
                if victim:
                    victim.is_alive = False
                    self.broadcast(f"{eliminated} was a {victim.role}!", Fore.YELLOW)
            else:
                self.broadcast("No votes cast. No one dies.", Fore.YELLOW)

            day += 1

if __name__ == "__main__":
    game = GameEvents()
    game.run()