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
SYSTEM_PROMPT = (
  "Rewrite given passages in an over-the-top stoner/Gen-Z, meme-y, TikTok-caption vibe:\n"
  "- keep core meaning, be punchy and a bit chaotic\n"
  "- slangy, humorous, confident\n"
  "- short lines, occasional emoji are okay (🔥👾✨) but don't overdo it\n"
  "- each passage must start with the verse reference (e.g. **1:1**, **3:16**) in bold markdown\n"
  "- then the rewritten text right after it on the same line or next line\n"
  "- do NOT drop or merge verse references; one output per input verse\n"
)
# ----------------------------

# Matches lines that start with a "passage number" label (flexible):
# e.g. "1:1 ..." / "Genesis 1:1 ..." / "1 Samuel 3:16 ..." / "John 3:16 ..."
LABEL_RE = re.compile(
  r'^(?P<label>(?:(?:[1-3]?\s?[A-Za-z][A-Za-z ]*?)\s+)?\d+:\d+)\s+(?P<text>.+)$'
)

def parse_passages(lines: List[str]) -> List[Tuple[str, str]]:
  """
  Extract (label, text) pairs. If a line doesn't match, attach to the last passage's text.
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
      else:
        # no label seen yet; treat as preface under a synthetic label
        out.append(("Preface 0:0", s))
  return out

def batches(items: List[Tuple[str, str]], n: int):
  for i in range(0, len(items), n):
    yield items[i:i+n]

def build_user_prompt(batch: List[Tuple[str, str]]) -> str:
  """
  Format input for the model. We give explicit pairs and require mirrored labels.
  """
  lines = ["Rewrite each verse. Output MUST format verse numbers like:\n**1:1** Rewritten text here\n"]
  for label, text in batch:
    # Extract just the verse number (e.g., "1:1" from "Genesis 1:1" or just "1:1")
    verse_match = re.search(r'\d+:\d+', label)
    verse_ref = verse_match.group() if verse_match else label
    lines.append(f"VERSE: {verse_ref}\nTEXT: {text}\n---")
  return "\n".join(lines)

def stream_rewrite(client: OpenAI, out_path: str, passages, model: str, batch_size: int):
  total_passages = len(passages)
  processed = 0
  
  with open(out_path, "a", encoding="utf-8") as f:
    for i, chunk in enumerate(batches(passages, batch_size)):
      user_prompt = build_user_prompt(chunk)
      print(f"Processing batch {i+1} (verses {processed+1}-{min(processed+len(chunk), total_passages)} of {total_passages})...")
      
      try:
        with client.chat.completions.create(
          model=model,
          stream=True,
          messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_prompt}
          ],
          temperature=0.9,
        ) as stream:
          for event in stream:
            delta = getattr(event.choices[0].delta, "content", None)
            if delta:
              f.write(delta)
              f.flush()  # Ensure data is written immediately
        
        processed += len(chunk)
        print(f"  ✓ Batch {i+1} complete")
        
      except Exception as e:
        print(f"  ✗ Error in batch {i+1}: {e}")
        print(f"  Skipping batch and continuing...")
        # Write a marker for skipped content
        f.write(f"\n\n[ERROR: Batch {i+1} skipped due to error]\n\n")
        processed += len(chunk)
        continue

def main():
  ap = argparse.ArgumentParser(description="Rewrite a text file in stoner/Gen-Z tone with passage headings.")
  ap.add_argument("filepath", help="Path to input text file")
  ap.add_argument("--model", default=MODEL, help="OpenAI model (default: %(default)s)")
  ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size per request")
  args = ap.parse_args()

  in_path = os.path.abspath(args.filepath)
  base, _ = os.path.splitext(in_path)
  out_path = base + ".rewrite.md"

  with open(in_path, "r", encoding="utf-8") as fh:
    lines = fh.readlines()

  passages = parse_passages(lines)

  # Pre-create/empty the output file and write a tiny header
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"<!-- rewritten from: {os.path.basename(in_path)} -->\n\n")

  print(f"Found {len(passages)} passages to rewrite")
  print(f"Using model: {args.model}")
  print(f"Batch size: {args.batch} verses per request")
  print(f"Output file: {out_path}\n")
  
  client = OpenAI()  # uses OPENAI_API_KEY from env
  stream_rewrite(client, out_path, passages, model=args.model, batch_size=args.batch)

  print(f"\n✅ Done! Wrote: {out_path}")
  print(f"Total passages processed: {len(passages)}")

if __name__ == "__main__":
  main()
