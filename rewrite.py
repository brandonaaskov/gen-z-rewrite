#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import List, Tuple
from openai import OpenAI

# --- config you can tweak ---
MODEL = "gpt-4o-mini"  # assume latest stable
BATCH_SIZE = 10        # verses/passages per API call (reduced for token limits)
DEFAULT_SYSTEM_PROMPT = (
  "Rewrite given text in an over-the-top stoner/Gen-Z, meme-y, TikTok-caption vibe:\n"
  "- keep core meaning, be punchy and a bit chaotic\n"
  "- slangy, humorous, confident\n"
  "- short lines, punchy paragraphs\n"
  "- maintain original paragraph structure\n"
  "- NO emojis unless specifically requested\n"
)

VERSE_SYSTEM_PROMPT = (
  "Rewrite given verses in an over-the-top stoner/Gen-Z, meme-y, TikTok-caption vibe:\n"
  "- keep core meaning, be punchy and a bit chaotic\n"
  "- slangy, humorous, confident\n"
  "- short lines\n"
  "- each verse must start with the verse reference (e.g. 1:1, 3:16)\n"
  "- then the rewritten text right after it\n"
  "- do NOT drop or merge verse references; one output per input verse\n"
  "- NO emojis unless specifically requested\n"
)

EMOJI_ADDON = "\n- add 0-2 relevant emojis at the end of each verse/paragraph (not more)"
# ----------------------------

# Matches lines that start with a "passage number" label (flexible):
# e.g. "1:1 ..." / "Genesis 1:1 ..." / "1 Samuel 3:16 ..." / "John 3:16 ..."
LABEL_RE = re.compile(
  r'^(?P<label>(?:(?:[1-3]?\s?[A-Za-z][A-Za-z ]*?)\s+)?\d+:\d+)\s+(?P<text>.+)$'
)

def detect_format(lines: List[str]) -> str:
  """
  Detect if input is verse-based or paragraph-based.
  Returns 'verse' if verse patterns found, 'paragraph' otherwise.
  """
  verse_pattern = re.compile(r'^\d+:\d+\s')
  verse_count = 0
  non_empty_lines = 0

  for line in lines:
    stripped = line.strip()
    if stripped:
      non_empty_lines += 1
      if verse_pattern.match(stripped):
        verse_count += 1

  # If more than 30% of non-empty lines start with verse pattern, treat as verse-based
  if non_empty_lines > 0 and verse_count / non_empty_lines > 0.3:
    return 'verse'
  return 'paragraph'

def parse_verses(lines: List[str]) -> List[Tuple[str, str]]:
  """
  Extract (label, text) pairs for verse-based content.
  """
  out: List[Tuple[str, str]] = []
  for line in lines:
    s = line.strip()
    if not s:
      continue
    m = LABEL_RE.match(s)
    if m:
      out.append((m.group("label").strip(), m.group("text").strip()))
    else:
      if out:
        # append continuation lines to the last passage text
        last_label, last_text = out[-1]
        out[-1] = (last_label, (last_text + " " + s).strip())
  return out

def parse_paragraphs(lines: List[str]) -> List[str]:
  """
  Extract paragraphs from regular text (split by blank lines).
  """
  paragraphs = []
  current = []

  for line in lines:
    stripped = line.strip()
    if not stripped:
      if current:
        paragraphs.append(' '.join(current))
        current = []
    else:
      current.append(stripped)

  # Add the last paragraph if exists
  if current:
    paragraphs.append(' '.join(current))

  return paragraphs

def batches(items: List[Tuple[str, str]], n: int):
  for i in range(0, len(items), n):
    yield items[i:i+n]

def build_verse_prompt(batch: List[Tuple[str, str]]) -> str:
  """
  Format input for verse-based content.
  """
  lines = ["Rewrite each verse. Output format: verse_number rewritten_text\n"]
  for label, text in batch:
    # Extract just the verse number (e.g., "1:1" from "Genesis 1:1" or just "1:1")
    verse_match = re.search(r'\d+:\d+', label)
    verse_ref = verse_match.group() if verse_match else label
    lines.append(f"VERSE: {verse_ref}\nTEXT: {text}\n---")
  return "\n".join(lines)

def build_paragraph_prompt(paragraphs: List[str]) -> str:
  """
  Format input for paragraph-based content.
  """
  return "Rewrite the following text, maintaining paragraph breaks:\n\n" + "\n\n".join(paragraphs)

def estimate_tokens(text: str) -> int:
  """Rough estimate: ~4 characters per token"""
  return len(text) // 4

def stream_rewrite(client: OpenAI, out_path: str, content, content_type: str, model: str, batch_size: int, use_emojis: bool):
  # Build system prompt
  if content_type == 'verse':
    system_prompt = VERSE_SYSTEM_PROMPT
  else:
    system_prompt = DEFAULT_SYSTEM_PROMPT

  if use_emojis:
    system_prompt += EMOJI_ADDON

  with open(out_path, "a", encoding="utf-8") as f:
    if content_type == 'verse':
      # Process verses in batches
      total_verses = len(content)
      processed = 0

      for i, chunk in enumerate(batches(content, batch_size)):
        user_prompt = build_verse_prompt(chunk)
        print(f"Processing batch {i+1} (verses {processed+1}-{min(processed+len(chunk), total_verses)} of {total_verses})...")

        try:
          with client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
              {"role":"system","content":system_prompt},
              {"role":"user","content":user_prompt}
            ],
            temperature=0.9,
          ) as stream:
            for event in stream:
              delta = getattr(event.choices[0].delta, "content", None)
              if delta:
                f.write(delta)
                f.flush()

          processed += len(chunk)
          print(f"  ✓ Batch {i+1} complete")

        except Exception as e:
          print(f"  ✗ Error in batch {i+1}: {e}")
          print(f"  Skipping batch and continuing...")
          f.write(f"\n\n[ERROR: Batch {i+1} skipped due to error]\n\n")
          processed += len(chunk)
          continue

    else:
      # Process paragraphs
      # Estimate tokens and decide batching
      full_text = "\n\n".join(content)
      estimated_tokens = estimate_tokens(full_text)

      if estimated_tokens < 2000:
        # Process all at once
        print(f"Processing {len(content)} paragraphs in a single batch...")
        user_prompt = build_paragraph_prompt(content)

        try:
          with client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
              {"role":"system","content":system_prompt},
              {"role":"user","content":user_prompt}
            ],
            temperature=0.9,
          ) as stream:
            for event in stream:
              delta = getattr(event.choices[0].delta, "content", None)
              if delta:
                f.write(delta)
                f.flush()
          print("  ✓ Complete")
        except Exception as e:
          print(f"  ✗ Error: {e}")
          f.write(f"\n\n[ERROR: Processing failed]\n\n")

      else:
        # Batch paragraphs (3-5 per batch depending on size)
        para_batch_size = 3 if estimated_tokens > 4000 else 5
        total_paras = len(content)
        processed = 0

        for i, chunk in enumerate(batches(content, para_batch_size)):
          print(f"Processing batch {i+1} (paragraphs {processed+1}-{min(processed+len(chunk), total_paras)} of {total_paras})...")
          user_prompt = build_paragraph_prompt(chunk)

          try:
            with client.chat.completions.create(
              model=model,
              stream=True,
              messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
              ],
              temperature=0.9,
            ) as stream:
              for event in stream:
                delta = getattr(event.choices[0].delta, "content", None)
                if delta:
                  f.write(delta)
                  f.flush()

            processed += len(chunk)
            print(f"  ✓ Batch {i+1} complete")
            if i < len(list(batches(content, para_batch_size))) - 1:
              f.write("\n\n")  # Add separator between batches

          except Exception as e:
            print(f"  ✗ Error in batch {i+1}: {e}")
            print(f"  Skipping batch and continuing...")
            f.write(f"\n\n[ERROR: Batch {i+1} skipped due to error]\n\n")
            processed += len(chunk)
            continue

def main():
  ap = argparse.ArgumentParser(description="Rewrite a text file in stoner/Gen-Z tone.")
  ap.add_argument("filepath", help="Path to input text file")
  ap.add_argument("--model", default=MODEL, help="OpenAI model (default: %(default)s)")
  ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size per request")
  ap.add_argument("--emojis", action="store_true", help="Add 0-2 emojis at the end of verses/paragraphs")
  args = ap.parse_args()

  in_path = os.path.abspath(args.filepath)

  # Output to rewritten directory with same filename
  base_filename = os.path.basename(in_path)
  base_name, _ = os.path.splitext(base_filename)
  out_path = os.path.join("./rewritten", base_name + ".rewrite.md")

  with open(in_path, "r", encoding="utf-8") as fh:
    lines = fh.readlines()

  # Detect format
  content_type = detect_format(lines)

  # Parse content based on detected format
  if content_type == 'verse':
    content = parse_verses(lines)
    content_description = f"{len(content)} verses"
  else:
    content = parse_paragraphs(lines)
    content_description = f"{len(content)} paragraphs"

  # Pre-create/empty the output file and write a tiny header
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"<!-- rewritten from: {os.path.basename(in_path)} -->\n\n")

  print(f"Detected format: {content_type}")
  print(f"Found {content_description} to rewrite")
  print(f"Using model: {args.model}")
  if args.emojis:
    print("Emojis: enabled (0-2 per unit)")
  print(f"Output file: {out_path}\n")

  client = OpenAI()  # uses OPENAI_API_KEY from env
  stream_rewrite(client, out_path, content, content_type, model=args.model, batch_size=args.batch, use_emojis=args.emojis)

  print(f"\n✅ Done! Wrote: {out_path}")
  print(f"Total {content_description} processed")

if __name__ == "__main__":
  main()
