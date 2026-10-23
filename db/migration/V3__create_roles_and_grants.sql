-- KN-VS-02: one database role per Knowledge workflow, with only the grants in spec Appendix B.
-- Login credentials / IAM mapping for these roles are provisioned outside the schema.
--
-- Invariant 9: nobody else gets a grant, so CONNECT is taken from PUBLIC and given back only
-- to these roles.

DO $$
DECLARE
    role_name text;
BEGIN
    FOREACH role_name IN ARRAY ARRAY['kn_ingest', 'kn_retrieve', 'kn_delete'] LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
            EXECUTE format('CREATE ROLE %I NOLOGIN', role_name);
        END IF;
        EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), role_name);
    END LOOP;
    EXECUTE format('REVOKE ALL ON DATABASE %I FROM PUBLIC', current_database());
    EXECUTE format(
        'GRANT CONNECT ON DATABASE %I TO kn_ingest, kn_retrieve, kn_delete', current_database());
END
$$;

-- The vector type and operators live in the public schema, so the roles need USAGE on it.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO kn_ingest, kn_retrieve, kn_delete;

REVOKE ALL ON kn_documents, kn_chunks FROM PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON kn_documents, kn_chunks TO kn_ingest;
GRANT SELECT ON kn_documents, kn_chunks TO kn_retrieve;
GRANT SELECT, DELETE ON kn_documents, kn_chunks TO kn_delete;
