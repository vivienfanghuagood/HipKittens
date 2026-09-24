# wx-ms-w7900d-0043 operating rules

The node is shared with other tenants and has been taken down once by a kernel of ours.
These are not style preferences.

## Node discipline

The node is shared and has been taken down once by a kernel.

- **GPUs 1/2/4 belong to other tenants. Use GPU 0 or 3 only**, pinned with
  `HIP_VISIBLE_DEVICES`.
- Compute the memory budget **before** spawning ranks. A cgroup OOM that
  interrupts a collective takes the whole machine down, not the process.
- Wrap every launch in an external `timeout --signal=KILL`, and bound every
  in-kernel wait (not with `clock64()`, see above).
- Never rebuild a `.so` while a process has it mapped.
- Serialise benchmark jobs; a co-tenant's load invalidates the numbers.
