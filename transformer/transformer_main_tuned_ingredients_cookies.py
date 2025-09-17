        # ----- Fixed anchor examples + a small rotating sample for qualitative sanity -----
        try:
            indices = list(anchor_idx)
            pool = [i for i in range(len(val_ds)) if i not in anchor_idx]
            rot_needed = max(0, cfg.LOG_EXAMPLES - len(indices))
            rng = random.Random(cfg.SHUFFLE_SEED + epoch)
            indices += rng.sample(pool, k=min(rot_needed, len(pool)))

            # --- NEW: ensure one example title contains "cookies" (case-insensitive) ---
            cookies_idx = None
            try:
                for ii, txt in enumerate(val_ds.df["input"].astype(str)):
                    if "cookies" in txt.lower():
                        cookies_idx = ii
                        break
            except Exception:
                cookies_idx = None

            if cookies_idx is not None:
                # Put the cookies example first and cap to LOG_EXAMPLES
                indices = [cookies_idx] + [i for i in indices if i != cookies_idx]
                if len(indices) > cfg.LOG_EXAMPLES:
                    indices = indices[:cfg.LOG_EXAMPLES]
            # --- END NEW ---

            for j, i in enumerate(indices, 1):
                title = val_ds.df.iloc[i]["input"]     # "recipe title" / input text
                gold_text = val_ds.df.iloc[i]["target"]
                pred_text = greedy_generate(model, sp, title, cfg.DEVICE, max_new_tokens=cfg.MAX_NEW_TOKENS)

                jac  = jaccard_from_texts(pred_text, gold_text)
                cos  = cosine_similarity_from_texts(pred_text, gold_text)
                rl   = rouge_l_f1_from_texts(pred_text, gold_text)
                bleu = bleu_from_texts(pred_text, gold_text, max_n=4)
                em   = 100.0 if pred_text.strip() == str(gold_text).strip() else 0.0

                print(f"  Example {j}:")
                print(f"    Title: {title}")
                print(f"    Pred:  {pred_text}")
                print(f"    Gold:  {gold_text}")
                print(f"    Metrics -> Jaccard: {jac:.2f}% | Cosine: {cos:.2f}% | ROUGE-L(F1): {rl:.2f}% | BLEU-4: {bleu:.2f}% | EM: {em:.2f}%")
        except Exception as e:
            print(f"(Skipping example print due to error: {e})")
