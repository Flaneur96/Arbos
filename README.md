# Arbos

![Arbos](arbos.jpg)

<p align="center">
  Welcome! Arbos is simply a <a href="https://ghuntley.com/loop/">Ralph-loop</a> combined with a Telegram bot.<br>
  That's all you need to do just about anything.
</p>

# The Design

```
                         (prompt.md + goal.md + state.md)
                       ┌─────────────────────────┐
                       ▼                         │
  ┌──────────┐     ┌───────┐                     │
  │ Telegram │◄───►│ Agent │─────────────────────┘
  └──────────┘     └───────┘
```

## Requirements

- [Telegram Bot token](https://core.telegram.org/bots#how-do-i-create-a-bot)
- [OpenRouter API key](https://openrouter.ai/settings/keys)

## Getting started

```sh
curl -fsSL https://raw.githubusercontent.com/unconst/Arbos/main/run.sh | bash
```

## Example

```
Make this my new goal:

Here are my credentials:
- Hyperliquid account key
- Coinglass API key
- LIUM compute API key

I would like you to use Timefm (Time Series Foundation Models) as the basis for your machine learning models. Please train these models online using real-time data from Coinglass, and backtest them on historical data. Leverage all available compute to run as many concurrent model architectures as possible.

Please ensemble the top-performing models, as determined by backtesting on real-time data, and use this ensemble to trade my Hyperliquid account.

Before beginning, research the current state of the art in crypto trading using reinforcement learning (RL) and machine learning (ML). Use your findings to improve the overall system design. Next, decompose this objective into actionable goals and implement the system accordingly.

After implementation, continuously monitor and proactively improve the system design and code. Regularly research, critique, and update your approach to ensure it remains truly state-of-the-art.
```

---

MIT 

