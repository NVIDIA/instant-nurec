# NVault (NVIDIA Vault as a Service)

## About NVault

NVault is NVIDIA's hosted secrets management service built on HashiCorp Vault. It provides a secure way to store, access, and manage sensitive credentials like passwords, API keys, SSH keys, and certificates.

## Key Features

- Namespaces: Isolated mini-Vaults for each team/org, managed by namespace admins via LDAP DL
- Auth Methods: Supports OIDC, AppRole, JWT, and Token authentication
- Secret Engines: KV (key-value) storage and dynamic secret generation
- NVault Agent/CLI: NVIDIA's custom Vault agent with GitLab/Jenkins integration, retries, and template rendering

For more information on NVault, please see [Nvault documentation](https://gitlab-master.nvidia.com/kaizen/services/vault/docs/-/blob/main/README.md?ref_type=heads&plain=0#kaizen-vault).

## Usage of NVault in NuRec

NuRec has a namespace `swtegra-nurec` provisioned in internal Staging and Production environemnta managed by the DL group `swtegra-nurec-vault-admins`. If you wish to manage NVault and be part of the DL group, please contact Leslie Meng at [lesliem@nvidia.com](mailto:lesliem@nvidia.com).

### To start using NVault cli:

Download NVault cli tool from [artifactory](https://urm.nvidia.com/artifactory/sw-kaizen-data-generic/com/nvidia/vault/vault-agent/), and add it to your system's PATH.

**Production:**

```bash
export VAULT_ADDR=https://prod.internal.vault.nvidia.com
export VAULT_NAMESPACE=swtegra-nurec
vault login -method="oidc" -path="oidc-admins" role="namespace-admin"
```

**Staging:**

```bash
export VAULT_ADDR=https://stg.internal.vault.nvidia.com
export VAULT_NAMESPACE=swtegra-nurec
vault login -method="oidc" -path="oidc-admins" role="namespace-admin"
```

> Note: You need to be part of the `swtegra-nurec-vault-admins` DL group to be able to log in with namespace-admin role.

### Common Commands

**Secrets**

```bash
# List all secret mounts
vault secrets list

# Read a secret
vault read <path/to/secret>

# Read a secret in JSON format
vault read -format=json <path/to/secret>

# Write/update a secret
vault write <path/to/secret> key1=value1 key2=value2
```

**Policies**

```bash
# List all policies
vault policy list

# Write/update a policy from file
vault policy write <policy-name> <policy-file.hcl>
```

**Token**

```bash
# Renew your token (before it expires)
vault token renew

# Renew with specific duration
vault token renew -increment=2h
```

## GitLab CI Integration

NuRec has integrated Vault secret fetching into GitLab CI for secure credential management. Currently, SSA client credentials for container scanning are fetched from Vault at runtime instead of being stored as CI/CD variables.

### How It Works

1. The `fetch-secrets` job authenticates to Vault using GitLab's JWT token
2. Vault agent runs with the config in `.gitlab/ci/vault/vault-agent.config` and renders the secrets template `.gitlab/ci/vault/vault-ci-secrets.tmpl`
3. Secrets are written to `my_secret.env` and uploaded as a dotenv artifact
4. Downstream jobs that declare `needs: [fetch-secrets]` automatically receive the exported variables
5. The dotenv artifact expires 5 minutes after the pipeline completes

> **Note**: JWT auth for GitLab CI is only configured in the **Production** Vault environment. It's recommended to store CI secrets in production to avoid additional setup. If you need to use the Staging environment, you'll need to configure JWT auth manually - see the [GitLab CI Vault Configuration Guide](https://gitlab-master.nvidia.com/kaizen/services/vault/docs/-/blob/main/guides/integrations/gitlab/gitlab-ci-vault-config.md).

### Adding New Secrets for CI

To add more secrets for GitLab CI to fetch from Vault:

1. **Store the secret in Vault** (requires namespace admin access):

   ```bash
   # For KV v2 (recommended - handles data/ prefix automatically)
   vault kv put <secret-engine-path>/<secret-name> key1=value1 key2=value2

   # For KV v1 or other engines
   vault write <secret-engine-path>/<secret-name> key1=value1 key2=value2
   ```

2. **Create a read-only policy** for CI to access the secret:

   ```bash
   vault policy write my-secret-ro - << EOF
   path "<secret-engine-path>/data/<secret-name>" {
     capabilities = ["read"]
   }
   EOF
   ```

   > **Note**: For KV v2, include `data/` in the path. For KV v1 or other engines (e.g., SSA), omit `data/`.

3. **Attach the policy to the JWT pipeline role**:

   ```bash
   # First, read existing policies (DO NOT override them)
   vault read auth/jwt/nvidia/gitlab-master/role/pipeline

   # Add your new policy to the existing list
   vault write auth/jwt/nvidia/gitlab-master/role/pipeline \
     policies="existing-policy-1,existing-policy-2,my-secret-ro"
   ```

   > ⚠️ **Important**: Always include all existing policies when updating. Omitting them will remove CI access to other secrets.

   For more details, see the [GitLab CI Vault Configuration Guide](https://gitlab-master.nvidia.com/kaizen/services/vault/docs/-/blob/main/guides/integrations/gitlab/gitlab-ci-vault-config.md).

4. **Update the secrets template** (`.gitlab/ci/vault/vault-ci-secrets.tmpl`):

   ```
   {{- with secret "nvidia/services/gitlab/pipelines/container-scans/ssa/prod/issue/creds" -}}
   SSA_CLIENT_ID={{ .Data.client_id }}
   SSA_CLIENT_SECRET={{ .Data.secret }}
   {{- end -}}
   {{- with secret "<secret-engine-path>/<secret-name>" -}}
   MY_NEW_VAR={{ .Data.data.<key> }}
   {{- end }}
   ```

   > **Template syntax notes:**
   >
   > - Use `{{-` and `-}}` to trim whitespace and avoid blank lines (GitLab dotenv format is strict)
   > - **KV v2** (most common): Use `.Data.data.<key>` to access secret values
   > - **KV v1 or other engines** (e.g., SSA): Use `.Data.<key>` directly
   >
   > To check your engine type, run `vault read -format=json <path>` and look at the JSON structure.

5. **Use the variable in your CI job**:
   - Add `needs: [fetch-secrets]` to your job
   - The variable `MY_NEW_VAR` will be automatically available as an environment variable

### Security Notes

- JWT auth ties Vault access to the specific `nre` GitLab project
- dotenv artifacts expire quickly (5 minutes after pipeline completion)
- dotenv artifacts don't appear in GitLab's artifact browser UI (unlike regular artifacts)
