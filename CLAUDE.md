# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based text rewriting tool that transforms biblical passages into Gen-Z/meme-style language using OpenAI's GPT models. The tool processes text files containing numbered verses (format: `\d+:\d+`) and outputs markdown files with bold verse references.

## Key Commands

### Running the rewriter
```bash
python rewrite.py <input_file> [--model MODEL] [--batch BATCH_SIZE]
```
- Default model: `gpt-4o-mini`
- Default batch size: 40 verses per API call
- Requires `OPENAI_API_KEY` environment variable

### Testing
```bash
python rewrite.py genesis.txt
```

## Architecture

### Core Components

- **rewrite.py**: Main script that handles verse parsing, batching, and API streaming
  - `parse_passages()`: Extracts verse number labels and text using regex pattern matching
  - `build_user_prompt()`: Formats batches for GPT with verse references extracted as `\d+:\d+`
  - `stream_rewrite()`: Streams responses from OpenAI API and writes to output file

### Input/Output Format

- **Input**: Text files with verses starting with numbers like `1:1`, `3:16` followed by text
- **Output**: Markdown files with verse numbers in bold (`**1:1**`) followed by rewritten text
- Output files named as `<input_base>.rewrite.md`

### Dependencies

- `openai` Python package for GPT API access
- Requires Python 3.x with type hints support