from captcha.image import ImageCaptcha
from pathlib import Path
import os

SAVE_ROOT = Path(__file__).parent / "dist"
os.makedirs(SAVE_ROOT, exist_ok=True)

SOURCE = "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
LENGTH = 4
REPEAT = 2

TOTAL = int( (len(SOURCE)**LENGTH)*REPEAT )

PRINT_EVERY = 10000

IC_GEN = ImageCaptcha()

if PRINT_EVERY % REPEAT != 0:
	raise Exception("Verbose unable to print")



def build_on(prefix, current, verbose, dry_run):
	if len(prefix) >= LENGTH:

		if verbose and (current % PRINT_EVERY)==0:
			print(f"  - Progress {100*current/TOTAL:.02f} % ({current}/{TOTAL})")#, end="\r")

		if not dry_run:
			for i in range(REPEAT):
				IC_GEN.write(
					prefix,
					SAVE_ROOT / f"{prefix}.{i}.png",
				)

		return prefix

	l = list()
	for c in SOURCE:
		r = build_on(prefix=prefix+c, current=current, verbose=verbose, dry_run=dry_run)
		if type(r) == str:
			l.append(
				r
			)
			current += REPEAT
		elif type(r) == list:
			l.extend(
				r
			)
			current += REPEAT * len(r)
		else:
			raise Exception(f"Unexpected return type from `build_on`: `{type(r)}`({r})")
	return l

def build(verbose=False, dry_run=False):
	current = 0
	current += REPEAT * len(build_on(
		prefix="",
		current=current,
		verbose=verbose,
		dry_run=dry_run,
	))
	if verbose:
		print(f"  - Progress {100*current/TOTAL:.02f} % ({current}/{TOTAL})")



if __name__ == "__main__":
	build(
		verbose=True,
		dry_run=False,
	)
