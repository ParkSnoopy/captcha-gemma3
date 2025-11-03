from captcha.image import ImageCaptcha
from pathlib import Path
import os
from threading import Thread
from time import sleep

SAVE_ROOT = Path(__file__).parent / "dist"
os.makedirs(SAVE_ROOT, exist_ok=True)

SOURCE = "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
LENGTH = 4
REPEAT = 2

TOTAL = int( (len(SOURCE)**LENGTH)*REPEAT )

PRINT_EVERY = 1000000

WORKERS = 20



class Worker:
	def __init__(self, master, pk):
		self.pk = pk
		self.master = master
		self.imggen = ImageCaptcha()

		self.print("Initialized")

	def print(self, msg):
		print(f"[W_{self.pk:02d}] {msg}")

	def do_work(self, data:str):
		#self.print(f"Working on: `{data}`")
		for i in range(REPEAT):
			self.imggen.write(
				data,
				SAVE_ROOT / f"{data}.{i}.png",
			)

	def run(self):
		self.print("Running")

		self._queue = list()
		while True:
			try:
				if len(self._queue) > 0:
					self.do_work(
						self._queue.pop()
					)
			except AttributeError:
				break

		self.print("Stopped")

	def stop(self):
		# Let `run` Panic
		self.print("Stopping...")
		del self._queue

class Master:
	def __init__(self):
		self.workers = [
			Worker(master=self, pk=i)
			for i in range(WORKERS)
		]
		self.workers_thread = list()

	def print(self, msg):
		print(f"[MNGR] {msg}")

	def status(self):
		#self.print("Current Status:")
		self.print(f"Assigned `{100*REPEAT*self._curr/TOTAL:.04f}`% :: ({TOTAL-self.pending_tasks()}/{TOTAL})")
		#for w in self.workers:
		#	print(f"    - Worker {w.pk:02d}: `{len(w._queue)}` pending")

	def assign_task(self, task:str):
		self.workers[ self._curr%WORKERS ]._queue.append(task)
		self._curr += 1

	def start(self):
		self._curr = 0
		self.workers_thread = [
			Thread(target=worker.run)
			for worker in self.workers
		]
		try:
			[
				w_thrd.start()
				for w_thrd in self.workers_thread
			]
			self.print("All Started")
		except Exception as exc:
			self.print(f"Panic Forced Stop: {exc}")
			[
				w.stop()
				for w in self.workers
			]

	def stop(self, _forced=False):
		try:
			if not _forced:
				while True:
					if self.pending_tasks() <= 0:
						break
					self.status()
					sleep(10)

			[
				w.stop()
				for w in self.workers
			]
			[
				w_thrd.join()
				for w_thrd in self.workers_thread
			]
			self.print("Normal Stop")
		except Exception as exc:
			self.print(f"Panic While Stopping: {exc}")
		self.print("All done")

	def pending_tasks(self) -> int:
		return REPEAT*sum([
			len(w._queue)
			for w in self.workers
		])


def build_on(prefix, current, verbose, dry_run, master):
	if len(prefix) >= LENGTH:

		if verbose and (current % PRINT_EVERY)==0:
			#print(f"  - Progress {100*current/TOTAL:.02f} % ({current}/{TOTAL})")#, end="\r")
			master.status()

		if not dry_run:
			master.assign_task(
				task=prefix,
				#current=current,
			)

		return prefix

	l = list()
	for c in SOURCE:
		r = build_on(prefix=prefix+c, current=current, verbose=verbose, dry_run=dry_run, master=master)
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

def build(verbose=False, dry_run=False, master=None):
	if (not dry_run) and (master == None):
		raise Exception("Must have `master` if not `dry_run`")

	if not dry_run:
		master.start()

	current = 0
	current += REPEAT * len(build_on(
		prefix="",
		current=current,
		verbose=verbose,
		dry_run=dry_run,
		master=master,
	))
	if verbose:
		#print(f"  - Progress {100*current/TOTAL:.02f} % ({current}/{TOTAL})")
		master.status()

	if not dry_run:
		master.stop()



if __name__ == "__main__":
	master = Master()

	try:
		build(
			verbose=True,
			dry_run=False,
			master=master,
		)
	except KeyboardInterrupt:
		master.stop(_forced=True)
