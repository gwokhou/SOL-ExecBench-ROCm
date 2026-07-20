# Reviewed SOLAR handlers

Formal analysis reads generated operation handlers only from this package
directory. `sol-execbench solar learn-handler` writes candidates elsewhere;
move a candidate here only after reviewing its source, verification evidence,
and content hashes, then commit all associated files together. Its handler JSON
must record `metadata.verification: passed`,
`metadata.formal_review: approved`, and the same `source_sha256` at both the
top level and in metadata. Formal loading recomputes the source digest and
rejects malformed, pending, tampered, missing, or path-traversing records.
