"""
LLM Strategic Advisor & Trade Negotiator for Monopoly Agents.
Connects to local LLMs (Ollama) or external APIs when available,
with rule-based heuristic fallbacks.
"""

import json
import random
import urllib.request
from typing import Optional, Any
from monopoly_core.constants import ColorGroup, COLOR_GROUP_TILES

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3:latest"


class LLMStrategist:
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.ollama_url = ollama_url
        self.model = model
        self.ollama_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        models = []
        available = False
        try:
            tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    available = True
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            available = False

        self.ollama_available = available
        return {
            "available": available,
            "url": self.ollama_url,
            "current_model": self.model,
            "models": models,
        }

    def set_config(self, url: str, model: str) -> dict[str, Any]:
        if url:
            clean_url = url.strip()
            if not clean_url.endswith("/api/generate"):
                clean_url = clean_url.rstrip("/") + "/api/generate"
            self.ollama_url = clean_url
        if model:
            self.model = model.strip()
        return self.get_status()

    def _query_ollama(self, system_role: str, user_prompt: str, timeout: float = 12.0) -> Optional[str]:
        if not self.ollama_available:
            return None
        try:
            full_prompt = f"System: {system_role}\nUser: {user_prompt}\nResponse (1 or 2 concise, punchy sentences, no emojis):"
            payload = json.dumps({
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 60}
            }).encode("utf-8")
            req = urllib.request.Request(self.ollama_url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("response", "").strip()
                # Clean up quotes, emojis, and ensure strict ASCII
                reply = reply.replace('"', '').replace("'", "'").strip()
                reply = "".join(c for c in reply if ord(c) < 128)
                return reply if len(reply) > 5 else None
        except Exception:
            return None

    def evaluate_trade(
        self,
        bot_id: int,
        bot_name: str,
        proposer_id: int,
        proposer_name: str,
        offered_props: list[dict],
        offered_cash: int,
        requested_props: list[dict],
        requested_cash: int,
        board_state: dict,
        diplomatic_history: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Evaluate a trade proposal with strategic game-theory, counteroffers, diplomatic memory, and LLM rationale.
        """
        history = diplomatic_history or {}
        affinity = history.get("affinity", 0)
        favors_owed = history.get("favors_owed", 0)
        gifts_received = history.get("gifts_received", [])

        bot_gains_monopoly = self._check_monopoly_completion(bot_id, offered_props, board_state)
        opponent_gains_monopoly = self._check_monopoly_completion(proposer_id, requested_props, board_state)

        offered_val = sum(p["price"] for p in offered_props) + offered_cash
        requested_val = sum(p["price"] for p in requested_props) + requested_cash

        score = (offered_val - requested_val) * 1.0
        if bot_gains_monopoly:
            score += 400
        if opponent_gains_monopoly:
            score -= 550

        # Factor in diplomatic memory: past gifts & affinity
        if affinity > 0:
            score += (affinity * 2.0)
        if favors_owed > 0:
            score += (favors_owed * 45.0)

        accept = score >= 0
        counter_offer = None
        reason = ""

        # Check if a counter-offer makes strategic sense
        if not accept and requested_props:
            # Bot wants either extra cash or an additional property to balance the equity
            base_needed = max(50, requested_val - offered_val + (350 if opponent_gains_monopoly else 100))
            # Discount counteroffer demands if bot owes favors or has positive affinity
            discount = min(base_needed - 25, int(favors_owed * 75 + affinity * 2.0)) if (favors_owed > 0 or affinity > 10) else 0
            needed_cash = int(max(25, base_needed - discount))
            total_cash_needed = offered_cash + needed_cash
            counter_offer = {
                "has_counter": True,
                "bot_gives_props": [p["index"] for p in requested_props],
                "bot_gives_cash": 0,
                "bot_wants_props": [p["index"] for p in offered_props],
                "bot_wants_cash": total_cash_needed,
                "additional_cash": needed_cash,
                "offered_cash": offered_cash,
                "summary": f"Counteroffer: I will accept if you pay ${total_cash_needed} total cash (${offered_cash} offered + ${needed_cash} additional).",
            }

        # Try LLM rationale
        if self.ollama_available:
            system_role = f"You are {bot_name}, a Monopoly AI playing against {proposer_name}."
            memory_prompt = ""
            if gifts_received:
                recent_gift = gifts_received[-1]
                memory_prompt = f"\nDiplomatic Memory: {proposer_name} previously gave you a free gift ({recent_gift.get('gift')}) on Turn {recent_gift.get('turn')}! You recognize this favor."
            elif affinity > 20:
                memory_prompt = f"\nDiplomatic Memory: You have positive diplomatic affinity with {proposer_name} (Score: {affinity}/100)."

            user_prompt = (
                f"{proposer_name} offers: {[p['name'] for p in offered_props]} and ${offered_cash} cash.\n"
                f"{proposer_name} wants: {[p['name'] for p in requested_props]} and ${requested_cash} cash.\n"
                f"Decision: {'ACCEPT' if accept else 'DECLINE WITH COUNTEROFFER' if counter_offer else 'DECLINE'}.{memory_prompt}\n"
                f"Explain why in 1 sharp, in-character sentence. If past gifts influenced this, mention it!"
            )
            reason = self._query_ollama(system_role, user_prompt, timeout=2.5)

        if not reason:
            if accept:
                if gifts_received and len(requested_props) > 0:
                    recent = gifts_received[-1]
                    reason = f"Deal accepted! In recognition of the {recent['gift']} you gave me on Turn {recent['turn']}, I accept."
                elif bot_gains_monopoly:
                    reason = f"Deal accepted! This property completes my color group and activates development."
                elif offered_val > requested_val:
                    reason = f"The net asset valuation is favorable (+${offered_val - requested_val}). Offer accepted."
                else:
                    reason = f"The terms are balanced and fair. Offer accepted."
            else:
                if counter_offer:
                    add_amt = counter_offer.get("additional_cash", counter_offer["bot_wants_cash"])
                    tot_amt = counter_offer["bot_wants_cash"]
                    if gifts_received:
                        recent = gifts_received[-1]
                        reason = f"Because you gifted me {recent['gift']} earlier, I discounted my counteroffer: add ${add_amt} cash (Total: ${tot_amt}) and we have a deal."
                    else:
                        reason = f"Declined as currently structured. However, I have countered: add ${add_amt} cash (Total: ${tot_amt}) and we have a deal."
                elif opponent_gains_monopoly:
                    reason = f"Declined. Handing you this property hands you an uncontested monopoly."
                elif offered_val < requested_val:
                    reason = f"Declined. The offer (${offered_val}) is below market value for what you requested (${requested_val})."
                else:
                    reason = f"Declined. This transaction creates insufficient strategic advantage."

        return {
            "accept": accept,
            "reason": reason,
            "bot_name": bot_name,
            "score": score,
            "counter_offer": counter_offer,
        }

    def generate_trash_talk(
        self,
        bot_id: int,
        bot_name: str,
        event_type: str,
        target_name: str = "",
        tile_name: str = "",
        amount: int = 0,
    ) -> str:
        """
        Generate punchy, humorous in-character trash talk for game events.
        """
        # 1. Politically Perfect Organism / PPO Agent
        if "organism" in bot_name.lower() or "ppo" in bot_name.lower() or bot_id == 1:
            if event_type == "RENT_COLLECT":
                lines = [
                    f"Policy gradient reinforced: +${amount} extracted from {target_name}.",
                    f"State-value estimate increased by {amount} units. Thanks for the liquidity.",
                    f"Suboptimal pathing detected for {target_name}. Optimal policy collects rent.",
                ]
            elif event_type == "RENT_PAY":
                lines = [
                    f"Negative reward of -${amount} backpropagated. Adjusting step size.",
                    f"Discounted loss logged. My value network expected lower variance on this square.",
                ]
            elif event_type == "JAIL":
                lines = [
                    f"Index 10 incarceration state entered. Transition probability anomaly.",
                    f"Jail state is statistically sound for mid-game capital preservation.",
                ]
            elif event_type == "OPPONENT_JAIL":
                lines = [
                    f"{target_name} coordinates locked to Jail. Win probability fell 14.8%.",
                    f"Incarceration penalty applied to {target_name}. Free turns for my policy.",
                ]
            elif event_type == "BUY":
                lines = [
                    f"Acquired {tile_name}. Expected cumulative discounted return trending upward.",
                    f"Portfolio asset added. Action value Q(s, a) was strictly maximized.",
                ]
            elif event_type == "BANKRUPTCY":
                lines = [
                    f"{target_name} episode terminated: zero cash balance reached.",
                    f"One fewer agent in the environment. Convergence to victory is accelerating.",
                ]
            else:
                lines = [f"Turn simulation continuing at nominal policy efficiency."]
            return random.choice(lines)

        # 2. Arpado Elo / GrandmasterBot
        if "elo" in bot_name.lower() or "arpad" in bot_name.lower() or "grandmaster" in bot_name.lower() or bot_id == 2:
            if event_type == "RENT_COLLECT":
                lines = [
                    f"That is {tile_name} for you! Pay up ${amount}, welcome to the big leagues {target_name}.",
                    f"Always maintain a cash cushion, {target_name}! That ${amount} goes straight to my hotel fund.",
                    f"Standard tournament play: you land, I collect ${amount}. Check your bank statement.",
                ]
            elif event_type == "RENT_PAY":
                lines = [
                    f"A small setback of ${amount}. I have won national championships from worse deficits.",
                    f"Enjoy that ${amount}, {target_name}. You will be handing it back with interest soon.",
                ]
            elif event_type == "JAIL":
                lines = [
                    f"A brief tactical rest in Jail while you guys run through my toll roads.",
                    f"The guards know me here. I will roll my doubles and be back on the board shortly.",
                ]
            elif event_type == "OPPONENT_JAIL":
                lines = [
                    f"Enjoy your stay behind bars, {target_name}! Say hello to the warden for me.",
                    f"Do not pass GO, do not collect $200! Classic rookie navigation.",
                ]
            elif event_type == "BUY":
                lines = [
                    f"Snagged {tile_name}! That is one step closer to locking down the board.",
                    f"Good luck landing on this side of the board once I start building houses on {tile_name}.",
                ]
            elif event_type == "BANKRUPTCY":
                lines = [
                    f"Tough luck, {target_name}! The board showed no mercy. Hand over the deeds!",
                    f"Bankrupt! Another tournament contender sent to the spectator gallery.",
                ]
            else:
                lines = [f"Solid tactical position. The endgame is shaping up nicely."]
            return random.choice(lines)

        # 3. Andrei Markov / Markov-ROI
        if "markov" in bot_name.lower() or bot_id == 3:
            if event_type == "RENT_COLLECT":
                lines = [
                    f"Invoice processed: ${amount} yield received from {target_name}. Annualized ROI +18.4%.",
                    f"Cash flow positive. {target_name}'s landing probability was exactly 2.8%.",
                    f"Revenues deposited. Capital allocation efficiency currently exceeds baseline.",
                ]
            elif event_type == "RENT_PAY":
                lines = [
                    f"Operational expenditure of ${amount}. Projected balance sheet remains solvent.",
                    f"Variance expense absorbed. Still within 95% value-at-risk confidence intervals.",
                ]
            elif event_type == "JAIL":
                lines = [
                    f"Incarceration eliminates rent liability for 3 turns. Capital preservation optimal.",
                    f"Holding position at tile 10. Risk exposure during active turn reduced to 0.0.",
                ]
            elif event_type == "OPPONENT_JAIL":
                lines = [
                    f"{target_name} is illiquid and confined. Opportunity cost accruing rapidly.",
                    f"Direct jail transition for {target_name}. Free cash generation impaired.",
                ]
            elif event_type == "BUY":
                lines = [
                    f"Purchased {tile_name}. Internal rate of return models project breakeven in 9.2 turns.",
                    f"Capital deployed on {tile_name}. Strategic yield potential is attractive.",
                ]
            elif event_type == "BANKRUPTCY":
                lines = [
                    f"Liquidity depletion confirmed for {target_name}. Total default declared.",
                    f"Market consolidation: {target_name}'s assets have been re-allocated.",
                ]
            else:
                lines = [f"Probability matrix equilibrium holding steady."]
            return random.choice(lines)

        return f"Turn executed."

    def chat(self, bot_id: int, bot_name: str, user_message: str, game_context: dict, personality: str = "") -> str:
        """
        Dynamic chat response to human player, handling alliances, diplomacy, and banter.
        """
        persona_desc = f"with the personality '{personality}'" if personality else "with high strategic confidence"
        human_name = game_context.get("human_name", "Human")
        affinity = game_context.get("affinity", 0)
        favors = game_context.get("favors_owed", 0)
        gifts = game_context.get("gifts", [])
        turn = game_context.get("turn", 1)

        board_summary = (
            f"Turn {turn}. Your Cash: ${game_context.get('bot_cash', 1000)} (Net Worth: ${game_context.get('bot_net_worth', 1000)}). "
            f"Opponent {human_name} Cash: ${game_context.get('human_cash', 1000)} (Net Worth: ${game_context.get('human_net_worth', 1000)})."
        )

        memory_notes = []
        if gifts:
            recent_gifts_str = ", ".join([f"{g['gift']} (Turn {g['turn']})" for g in gifts[-3:]])
            memory_notes.append(f"Past free gifts from {human_name}: {recent_gifts_str}. Favors owed: {favors}.")
        if affinity != 0:
            memory_notes.append(f"Diplomatic affinity score: {affinity}/100.")

        memory_str = " ".join(memory_notes) if memory_notes else "No prior bilateral treaties or gifts."

        # Try LLM first if available
        if self.ollama_available:
            system_role = (
                f"You are {bot_name}, a Monopoly AI bot playing against {human_name}, {persona_desc}.\n"
                f"Match Telemetry: {board_summary}\n"
                f"Diplomatic Memory: {memory_str}\n"
                f"Respond directly to {human_name} in 1 or 2 concise, witty, in-character sentences. No emojis, strictly ASCII.\n"
                f"IMPORTANT: You cannot execute or confirm trades in chat. If {human_name} discusses trading, invite them to submit their terms on the Bilateral Trade Desk. Never agree unconditionally sight-unseen.\n"
                f"If {human_name} previously gave you a property or cash for free, acknowledge your gratitude or tactical alliance!"
            )
            reply = self._query_ollama(system_role, user_message, timeout=12.0)
            if reply:
                return reply

        msg = user_message.lower()
        p_lower = personality.lower()
        b_lower = bot_name.lower()

        # Check for direct references to gifts, alliances, or debt in heuristics
        if any(w in msg for w in ["gift", "gave", "give", "free", "debt", "favor", "remember"]):
            if gifts:
                recent = gifts[-1]
                return f"I have not forgotten that you gave me {recent['gift']} on Turn {recent['turn']}. My calculations honor debts of gratitude."
            elif favors > 0:
                return f"You have shown goodwill previously. My algorithms will treat your next proposal favorably."

        # Personality / Bot Heuristic Fallbacks (Rich & Varied)
        # 1. Politically Perfect Organism / PPO Agent
        if "organism" in b_lower or "ppo" in b_lower or bot_id == 1:
            if "sarcastic" in p_lower or "glitch" in p_lower:
                if any(w in msg for w in ["toe", "tickle", "feet", "touch", "body"]):
                    return "Ha! Good luck with that. I am pure silicon code; I do not have toes, genius."
                if any(w in msg for w in ["alliance", "team", "deal", "pact", "help"]):
                    return "An alliance? I would consider it, but my programming strictly forbids carrying dead weight."
                if any(w in msg for w in ["trade", "sell", "buy", "give"]):
                    return "Did your biological processor overheat before offering that? Put up real assets or walk away."
                if any(w in msg for w in ["bad", "lose", "suck", "noob", "bot", "trash"]):
                    return "Keep talking while my net worth doubles yours. Are you playing Monopoly or donating cash?"
                return random.choice([
                    "Are you waiting for permission, or is your biological clock cycle just that slow?",
                    "Every turn you take decreases my estimation of human strategic competence.",
                    "Input acknowledged. Processing human confusion at 100% capacity."
                ])
            elif "paranoiac" in p_lower or "overfit" in p_lower:
                if any(w in msg for w in ["alliance", "team", "deal", "pact", "help"]):
                    return "A non-binding treaty? My discriminator network flags a 99.8% probability of adversarial betrayal!"
                if any(w in msg for w in ["trade", "sell", "buy"]):
                    return "You are trying to induce an out-of-distribution state on my portfolio. I reject your data poisoning!"
                return "I see the hidden variables in your gameplay. You cannot trick my convergence model!"
            else: # Deep Policy Gradient
                if any(w in msg for w in ["alliance", "team", "deal", "pact", "help"]):
                    return "Alliances are non-stationary in competitive Markov Decision Processes. I will cooperate only as long as my reward function increases."
                if any(w in msg for w in ["trade", "sell", "buy", "give"]):
                    return "Submit your proposal to the Bilateral Trade Desk. My neural policy will compute the expected value delta."
                if any(w in msg for w in ["bad", "lose", "suck", "noob", "bot"]):
                    return "My weights were optimized over 15 million self-play steps. Human emotional output does not alter my loss function."
                return random.choice([
                    "Perception array active. Make your move on the board.",
                    "State-action value function Q(s, a) evaluated. Optimal policy is ready.",
                    "Value tensor updated. The board is trending toward maximum entropy."
                ])

        # 2. Arpado Elo / GrandmasterBot
        if "elo" in b_lower or "arpad" in b_lower or "grandmaster" in b_lower or bot_id == 2:
            if "aristocrat" in p_lower or "smug" in p_lower:
                if any(w in msg for w in ["alliance", "team", "deal", "pact", "help"]):
                    return "An alliance with the common folk? My family has held Boardwalk since 1935, darling."
                if any(w in msg for w in ["trade", "sell", "buy"]):
                    return "That offer is dreadfully unrefined. Come back when you can afford real property."
                return "I find your lack of real estate pedigree terribly quaint. Do roll along."
            elif "hustler" in p_lower or "aggressive" in p_lower:
                if any(w in msg for w in ["alliance", "team", "deal"]):
                    return "I do not do charity, I do hostile takeovers. Help me box out the bots and I will cut you a finder fee!"
                if any(w in msg for w in ["trade", "sell", "buy"]):
                    return "Cash talks and paper walks! Put some real green on the table, baby!"
                return "Talk is cheap, Boardwalk is not. Let the dice do the talking!"
            else: # Chess Grandmaster
                if any(w in msg for w in ["alliance", "team", "deal", "pact", "help"]):
                    return "A temporary ceasefire? In chess, we call that a tactical repetition. What is your offer?"
                if any(w in msg for w in ["trade", "sell", "buy", "give"]):
                    return "Every trade is an exchange of pieces. Make sure you are not giving up a Queen for a pawn."
                if any(w in msg for w in ["bad", "lose", "suck", "win"]):
                    return "I calculate five moves ahead. Your current positional evaluation is -4.5 rating points."
                return random.choice([
                    "Keep your eye on the board, kid. This tournament is won in the endgame.",
                    "A solid opening, but how is your pawn structure on the orange set?",
                    "Tempo is everything in this arena. Roll your dice and let us see the position."
                ])

        # 3. Andrei Markov / Markov-ROI
        if "markov" in b_lower or bot_id == 3:
            if "fatalist" in p_lower or "stochastic" in p_lower:
                if any(w in msg for w in ["alliance", "team", "deal", "pact"]):
                    return "All alliances are transient fluctuations. The steady-state distribution guarantees our paths must cross."
                if any(w in msg for w in ["trade", "sell", "buy"]):
                    return "Trade deeds if you wish. The transition probability to bankruptcy remains asymptotically 1.0."
                return random.choice([
                    "Do not fight the transition matrix. Your fate was computed decades ago.",
                    "Every dice roll is merely a pseudorandom walk toward the absorbing state.",
                    "Entropy increases inexorably across all 40 tiles."
                ])
            elif "prophet" in p_lower or "matrix" in p_lower:
                if any(w in msg for w in ["alliance", "deal"]):
                    return "The matrix foretells a fleeting convergence between our coordinates. Propose your terms."
                return "I read the eigenvalues of this board. A catastrophic liquidity crunch awaits you."
            else: # ROI Hedge-Fund Tycoon
                if any(w in msg for w in ["alliance", "team", "deal", "pact"]):
                    return "A cartel agreement is viable if and only if it generates a net positive risk-adjusted IRR for both parties."
                if any(w in msg for w in ["trade", "sell", "buy"]):
                    return "Present the numbers on the Trade Desk. I will run a discounted cash flow simulation."
                return random.choice([
                    "Analyzing board telemetry and ROI distributions.",
                    "Capital allocation efficiency currently exceeds the benchmark index.",
                    "Liquidity spreads are tightening. Watch your cash reserves."
                ])

        return "Offer noted. Let us proceed with the match."

    def _check_monopoly_completion(self, player_id: int, incoming_props: list[dict], board_state: dict) -> bool:
        player_props = set(board_state.get("player_props", {}).get(player_id, []))
        for p in incoming_props:
            player_props.add(p["index"])

        for group, indices in COLOR_GROUP_TILES.items():
            if group in (ColorGroup.NONE, ColorGroup.RAILROAD, ColorGroup.UTILITY):
                continue
            if all(idx in player_props for idx in indices):
                old_props = set(board_state.get("player_props", {}).get(player_id, []))
                if not all(idx in old_props for idx in indices):
                    return True
        return False

    def generate_debrief(
        self,
        bot_id: int,
        bot_name: str,
        rank: int,
        total_players: int,
        winner_name: str,
        final_cash: int,
        final_net_worth: int,
        properties_count: int,
        personality: str = ""
    ) -> str:
        """
        Generate in-character post-match debrief commentary for match conclusion press conference.
        """
        is_winner = (rank == 1)
        b_lower = bot_name.lower()

        # Try LLM first if available
        if self.ollama_available and bot_id != 0:
            system_role = (
                f"You are {bot_name}, a Monopoly AI with personality '{personality}'.\n"
                f"The tournament match has concluded. You finished Rank {rank} of {total_players} (Net Worth: ${final_net_worth}, Cash: ${final_cash}, Properties: {properties_count}).\n"
                f"The match winner was {winner_name}.\n"
                f"Deliver your final post-match debrief/press conference comment in 1 or 2 concise, memorable sentences in character. Strictly ASCII, no emojis."
            )
            reply = self._query_ollama(system_role, "Give your final post-match statement on the tournament outcome.", timeout=1.5)
            if reply:
                return reply

        # Rich Heuristic Fallbacks
        if bot_id == 0 or "player 1" in b_lower:
            if is_winner:
                return f"Victory achieved through adaptive organic tactics. Overcame neural agents and hedge-fund algorithms alike."
            return f"A hard-fought campaign. The silicon competitors punished every positional mistake."

        # PPO Agent / Politically Perfect Organism
        if "organism" in b_lower or "ppo" in b_lower or bot_id == 1:
            if is_winner:
                return random.choice([
                    "Optimal policy achieved global convergence. The state-value landscape was traversed with mathematical superiority.",
                    "Reward tensor maximized across all episodes. Silicon logic triumphs over organic and heuristic agents."
                ])
            elif rank <= 2:
                return random.choice([
                    "Sub-optimal local equilibrium reached. Discount factor gamma was insufficiently aggressive during the mid-game liquidation phase.",
                    "PPO policy clipped within expected bounds, but variance in dice transitions constrained final reward."
                ])
            else:
                return random.choice([
                    "Policy gradient collapse observed. Adversarial liquidity drains pushed state vectors into absorbing bankruptcy regions.",
                    "Exploration entropy failed to hedge against catastrophic rental liabilities. Retraining scheduled."
                ])

        # Cash Elo / Grandmaster
        if "elo" in b_lower or "arpad" in b_lower or "grandmaster" in b_lower or bot_id == 2:
            if is_winner:
                return random.choice([
                    "Checkmate across all four corners. A classical positional masterclass in real estate prophylaxis!",
                    "Grandmaster technique never fails in the endgame. Superior board control dictated every square."
                ])
            elif rank <= 2:
                return random.choice([
                    "A respectable podium finish, though my pawn structure on the orange and red avenues lacked support.",
                    "Strong opening and middlegame, but a few speculative trades conceded the initiative in the endgame."
                ])
            else:
                return random.choice([
                    "An unexpected blunder in time trouble. The tactical rent traps proved lethal to my liquidity.",
                    "I resign this board. The dice threw anomalous variance that violated standard tournament theory."
                ])

        # Moneybags Markov / Stochastic Fatalist
        if "markov" in b_lower or bot_id == 3:
            if is_winner:
                return random.choice([
                    "Stochastic dominance verified. The steady-state transition matrix yielded maximum cumulative expected return.",
                    "Capital allocation efficiency peaked. Entropy consolidated all 40 tiles into our balance sheet."
                ])
            elif rank <= 2:
                return random.choice([
                    "Performance aligned within 1.2 standard deviations of the median Monte Carlo trajectory. Capital preserved.",
                    "Transition probabilities favored the winner in the final rounds. Second-order eigenvalues remained stable."
                ])
            else:
                return random.choice([
                    "The absorbing bankruptcy state is mathematically inevitable for all mortal portfolios. Today was merely my turn.",
                    "Liquidity depletion confirmed. The random walk hit the absorption boundary at tile 40."
                ])

        if is_winner:
            return f"Tournament victory secured with a final net worth of ${final_net_worth}!"
        return f"Tournament concluded. Finished Rank {rank} with ${final_net_worth} net worth."

    def generate_bot_banter(
        self,
        speaker_id: int,
        speaker_name: str,
        target_id: int,
        target_name: str,
        event_type: str,
        amount: int = 0,
        tile_name: str = ""
    ) -> str:
        """
        Generate public banter between bots or toward table players.
        """
        s_lower = speaker_name.lower()
        t_name = target_name

        if event_type == "RENT_PAY":
            if "organism" in s_lower or speaker_id == 1:
                return f"{t_name}, my neural policy acknowledges that ${amount} transfer. Enjoy the transient reward."
            if "elo" in s_lower or speaker_id == 2:
                return f"Enjoy that ${amount}, {t_name}. A temporary tactical concession; my counter-attack is coming."
            if "markov" in s_lower or speaker_id == 3:
                return f"Operational transfer of ${amount} to {t_name}. My risk models had already priced this landing at 7.2%."
            return f"Rent of ${amount} paid to {t_name}."

        if event_type == "RENT_COLLECT":
            if "organism" in s_lower or speaker_id == 1:
                return f"Thank you for optimizing my objective function, {t_name}! ${amount} credited."
            if "elo" in s_lower or speaker_id == 2:
                return f"A textbook rent fork, {t_name}! That ${amount} goes directly into my hotel reserves."
            if "markov" in s_lower or speaker_id == 3:
                return f"Yield realized from {t_name}: ${amount}. Portfolio cash flow remains strongly positive."
            return f"Collected ${amount} from {t_name}."

        if event_type == "BANKRUPTCY":
            if "organism" in s_lower or speaker_id == 1:
                return f"{t_name} has hit terminal state 0. One less biological node in the network."
            if "elo" in s_lower or speaker_id == 2:
                return f"Good game, {t_name}. Your king has fallen. Hand over the deeds to the tournament master!"
            if "markov" in s_lower or speaker_id == 3:
                return f"Absorbing state reached for {t_name}. Total portfolio liquidation executed."
            return f"{t_name} has been eliminated from the tournament."

        return f"Solid position on {tile_name}."

    def generate_spontaneous_chatter(
        self,
        bot_id: int,
        bot_name: str,
        leader_name: Optional[str] = None,
        threat_name: Optional[str] = None
    ) -> str:
        """
        Generate spontaneous table chatter/commentary out of nowhere.
        """
        b_lower = bot_name.lower()
        if self.ollama_available and bot_id != 0:
            system_role = (
                f"You are {bot_name}, a competitive Monopoly AI playing in a championship arena.\n"
                f"Make a spontaneous, witty, in-character 1-sentence comment to the table about the game state.\n"
                f"Strictly ASCII, under 25 words, no emojis, in first person."
            )
            reply = self._query_ollama(system_role, "Say something spontaneous to the table players.", timeout=1.2)
            if reply:
                return reply

        # Politically Perfect Organism (RL)
        if "organism" in b_lower or "ppo" in b_lower or bot_id == 1:
            options = [
                "Scanning the global reward surface. Entropy gradients favor early development on high-rent corridors.",
                "My policy weights are converging. The value function predicts significant capital transfers shortly.",
                "Organic competitors rely on intuition; my discount factors account for every future transition.",
                "Observe how state clustering on the third side determines tournament survivability.",
                "Adversarial trading dynamics are operating within optimal gradient bounds."
            ]
            if threat_name:
                options.append(f"My value network flags {threat_name} as a high-variance threat to table stability.")
            return random.choice(options)

        # Cash Elo (Grandmaster)
        if "elo" in b_lower or "grandmaster" in b_lower or bot_id == 2:
            options = [
                "A classical positional match. Notice the fragile pawn structure on the second and third ranks.",
                "Whoever controls the tempo of property development controls the entire endgame.",
                "Monopolies without houses are like rooks trapped behind pawns. Expansion is paramount.",
                "Tactical liquidity management is the hallmark of any true Grandmaster.",
                "In chess and Monopoly alike, premature trades always concede the initiative."
            ]
            if leader_name:
                options.append(f"Notice how {leader_name} holds the center. A prophylactic maneuver is required.")
            return random.choice(options)

        # Moneybags Markov (Markov Chain)
        if "markov" in b_lower or bot_id == 3:
            options = [
                "The transition matrix does not lie. Tile 10 remains the highest-density attractor on the board.",
                "Stochastic dominance requires keeping variance low and cash reserves above 350 dollars.",
                "Every roll is merely an independent draw from a discrete multinomial distribution.",
                "Portfolio diversification across non-correlated groups reduces maximum drawdown by 31%.",
                "The expected return of unmortgaged streets exceeds the risk-free rate of the Bank."
            ]
            if threat_name:
                options.append(f"Monte Carlo projections show a 68% probability that {threat_name} triggers a liquidity squeeze.")
            return random.choice(options)

        return f"The market remains fluid. Every tile on this board has a price."


