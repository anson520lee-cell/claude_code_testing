# Open the terminal from any device (one computer runs it, the others only browse)

Goal: one always-on computer (the Windows PC) runs everything - the daily analysis, the intraday price task
and the dashboard server. The Mac, a phone or a tablet only open a web address such as
`https://swing.tail1a2b3c.ts.net` (or just `http://swing`) instead of `http://127.0.0.1:8000`.

## Recommended route: Tailscale (free, private)

[Tailscale](https://tailscale.com) puts your own devices on a private network. The dashboard is **not**
on the public internet: only devices signed in to your Tailscale account can open it, with no router
settings and no open ports. The free Personal plan is enough (up to 3 users, 100 devices).

What you get:

| Address | Works on |
|---|---|
| `https://swing.<your-tailnet>.ts.net` | every device in your tailnet (real HTTPS certificate) |
| `http://swing` | every device in your tailnet with MagicDNS on (the default) |

`swing` is the machine name you give the Windows PC in Tailscale; `<your-tailnet>` is a name Tailscale
assigns (you can switch it to a friendlier generated name under DNS -> Tailnet name in the admin console).

How it is wired: `serve.py` keeps listening on `127.0.0.1:8000` only (nothing else on your home network can
reach it). Tailscale Serve, running on the same PC, accepts connections from your tailnet on ports 443 / 80
and passes them to `127.0.0.1:8000`. The dashboard still asks for the password (`serve.py --set-password`),
so a lost or borrowed device in your tailnet cannot edit the journal or alerts without it.

Setup on Windows: `docs/WINDOWS.md` -> "Always-on host" (`scripts\setup_windows_host.ps1` does the PC part).
Setup on the Mac: install Tailscale, sign in with the same account, then
`bash scripts/mac_browser_only.sh https://swing.<your-tailnet>.ts.net` (stops the Mac's own jobs and points
the Dock app at the PC). Phone: install the Tailscale app, sign in, open the address.

## Other free options (not used)

| Option | Name you get | Reachable by | Why not |
|---|---|---|---|
| Tailscale Funnel (`tailscale funnel --bg 8000`) | the same `https://swing.<tailnet>.ts.net` | anyone on the internet, no app needed | public: only the password protects your journal; use it only if a device cannot run Tailscale |
| Cloudflare Tunnel | your own domain (`terminal.example.com`) | anyone on the internet | needs a domain you own on Cloudflare (the free `trycloudflare.com` names are random and change every restart); public unless you also set up Cloudflare Access |
| Router port forward + free DDNS (DuckDNS, No-IP) | `something.duckdns.org` | anyone on the internet | opens a port on your home IP, router setup, self-managed HTTPS; the riskiest |

## Login

* `python serve.py --set-password` (Windows: `.venv\Scripts\python.exe serve.py --set-password`) asks for
  a password twice. Only a salted PBKDF2 hash is stored, in `data\auth.json` (git-ignored, never uploaded).
* The password is asked every time you open the dashboard in a new browser session: the login is a
  session cookie (HttpOnly, SameSite=Lax, no expiry date), so closing the browser signs you out. It also
  signs out after 30 minutes without activity and after 12 hours at most. Visit `/logout` to sign out
  sooner. Changing the password signs every device out.
* Five wrong passwords from one address lock it out for 15 minutes; 30 from all addresses together lock
  the login for everyone for 15 minutes (slows down guessing if the address is ever public).
* Requests made on the PC itself straight to `http://127.0.0.1:8000` do not need the password (the
  launcher uses this). Anything relayed by Tailscale Serve, Funnel or another proxy carries forwarding
  headers and must sign in.
* `serve.py` refuses `--host 0.0.0.0` (or any non-local address) until a password is set.
* `python serve.py --clear-password` removes the login.
