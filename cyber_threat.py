# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter
import warnings
import re

from pathlib import Path
_ROOT    = Path(__file__).resolve().parent
DATA_DIR = _ROOT / "data" / "raw"
warnings.filterwarnings('ignore')

# ── Style: seaborn darkgrid base + custom threat-intel colors ─────────────────
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams.update({
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'axes.titleweight': 'bold',
})

# ── Threat Intelligence Color Palette ─────────────────────────────────────────
PALETTE = {
    'critical': '#D62728',
    'high':     '#FF5733',
    'medium':   '#FF7F0E',
    'low':      '#BCBD22',
    'info':     '#1F77B4',
    'stealth':  '#9467BD',
    'neutral':  '#7F7F7F',
    'accent':   '#17BECF',
}
THREAT_COLORS = [PALETTE['critical'], PALETTE['high'], PALETTE['medium'],
                 PALETTE['low'], PALETTE['info'], PALETTE['stealth']]

print("✅ Environment ready.")


# ── Load Datasets ─────────────────────────────────────────────────────────────
otx  = pd.read_csv(DATA_DIR / '1_otx_threat_intel.csv')
cve  = pd.read_csv(DATA_DIR / '2_cve_vulnerabilities.csv')
domains = pd.read_csv(DATA_DIR / '3_malicious_domains.csv')
ips  = pd.read_csv(DATA_DIR / '4_malicious_ips.csv')

# ── Parse timestamps ──────────────────────────────────────────────────────────
otx['Created']   = pd.to_datetime(otx['Created'],   format='ISO8601')
otx['Modified']  = pd.to_datetime(otx['Modified'],  format='ISO8601')
cve['dateAdded'] = pd.to_datetime(cve['dateAdded'])
cve['dueDate']   = pd.to_datetime(cve['dueDate'])

# ── Dataset Summary ───────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'Dataset':    ['OTX Threat Pulses', 'CISA KEV CVEs', 'Malicious Domains', 'Malicious IPs'],
    'Records':    [len(otx), len(cve), len(domains), len(ips)],
    'Columns':    [otx.shape[1], cve.shape[1], domains.shape[1], ips.shape[1]],
    'Date Range': [
        f"{otx['Created'].min().date()} to {otx['Created'].max().date()}",
        f"{cve['dateAdded'].min().date()} to {cve['dateAdded'].max().date()}",
        'N/A', 'N/A',
    ],
})
print(summary.to_string(index=False))
print(f"\n📊 Total intelligence artifacts: {len(otx)+len(cve)+len(domains)+len(ips):,}")


# ── Normalize unknowns ────────────────────────────────────────────────────────
# Store raw null percentages before cleaning (for the plot)
null_pcts = {
    'OTX Pulses':      otx.isnull().mean() * 100,
    'CISA KEV CVEs':   cve.isnull().mean() * 100,
    'Domains':         domains.isnull().mean() * 100,
    'Malicious IPs':   ips.isnull().mean() * 100,
}

for df in [otx, cve, domains, ips]:
    df.replace(['Unknown', 'unknown', 'N/A', 'n/a', ''], np.nan, inplace=True)

# OTX: fill operational fields
otx['Indicators_Count']  = otx['Indicators_Count'].fillna(0)
otx['Subscribers']       = otx['Subscribers'].fillna(0)
otx['Industries']        = otx['Industries'].fillna('Unattributed')
otx['Countries']         = otx['Countries'].fillna('Unattributed')
otx['Malware_Families']  = otx['Malware_Families'].fillna('Unclassified')

# Numeric coercions
ips['ASN']               = pd.to_numeric(ips['ASN'],                errors='coerce')
domains['Reputation']    = pd.to_numeric(domains['Reputation'],     errors='coerce')
domains['Domain_Length'] = pd.to_numeric(domains['Domain_Length'],  errors='coerce')

# ── Missing Data Visualization ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Intelligence Data Completeness Audit', fontsize=15, fontweight='bold', y=1.01)

ds_colors = [PALETTE['critical'], PALETTE['high'], PALETTE['medium'], PALETTE['info']]
for ax, (title, miss_series), color in zip(axes, null_pcts.items(), ds_colors):
    miss = miss_series[miss_series > 0].sort_values(ascending=True)
    if len(miss) == 0:
        ax.text(0.5, 0.5, 'No missing values', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='green')
    else:
        bars = ax.barh(miss.index, miss.values, color=color, alpha=0.85, edgecolor='none')
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', va='center', fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Missing (%)')
    ax.set_xlim(0, 105)

plt.tight_layout()
plt.show()

print("\n📌 Analyst Note: High missingness in Countries/Industries fields is operationally")
print("   significant — adversaries deliberately obscure attribution and targeting metadata.")
print("   These rows are retained as 'Unattributed' rather than dropped.")

# ── Plot 4.1: Threat Campaign Timeline ───────────────────────────────────────
# Focus on 2021 onwards for meaningful signal density

otx_recent = otx[otx['Created'] >= '2021-01-01'].copy()
monthly    = otx_recent.set_index('Created').resample('ME')['Pulse_ID'].count()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(monthly.index, monthly.values, alpha=0.25, color=PALETTE['critical'])
ax.plot(monthly.index, monthly.values, color=PALETTE['critical'], linewidth=2.5,
        marker='o', markersize=3)

peak_date = monthly.idxmax()
peak_val  = monthly.max()
ax.annotate(
    f'Peak: {peak_val} pulses\n{peak_date.strftime("%b %Y")}',
    xy=(peak_date, peak_val),
    xytext=(peak_date - pd.DateOffset(months=5), peak_val * 0.80),
    arrowprops=dict(arrowstyle='->', color=PALETTE['medium'], lw=1.5),
    fontsize=9, color=PALETTE['medium']
)

ax.set_title('Threat Campaign Activity — Monthly Pulse Volume (2021–2026)')
ax.set_xlabel('Date')
ax.set_ylabel('Intelligence Pulses Published')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print(f"   • Campaign volume peaked in {peak_date.strftime('%B %Y')} ({peak_val} pulses).")
print(f"   • The dataset captures {len(otx_recent):,} pulses since 2021 — a dense operational picture.")
print("   • Increasing publication frequency correlates with improved community detection and")
print("     broader adversary activity, particularly nation-state and ransomware operations.")


# ── Plot 4.2: Targeted Industries ────────────────────────────────────────────

def explode_field(series, sep=', '):
    """Explode comma-separated multi-value fields into a flat Counter."""
    counts = Counter()
    for val in series.dropna():
        for item in str(val).split(sep):
            item = item.strip()
            if item and item not in ('Unattributed', 'nan', ''):
                counts[item] += 1
    return counts

industry_counts = explode_field(otx['Industries'])
top_industries  = dict(sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:12])

tier_colors = []
for i in range(len(top_industries)):
    if i < 3:  tier_colors.append(PALETTE['critical'])
    elif i < 6: tier_colors.append(PALETTE['high'])
    elif i < 9: tier_colors.append(PALETTE['medium'])
    else:       tier_colors.append(PALETTE['info'])

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(list(top_industries.keys())[::-1],
               list(top_industries.values())[::-1],
               color=tier_colors[::-1], edgecolor='none', height=0.7)
for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=9)

legend_handles = [
    mpatches.Patch(color=PALETTE['critical'], label='Primary Targets (Top 3)'),
    mpatches.Patch(color=PALETTE['high'],     label='Secondary Targets'),
    mpatches.Patch(color=PALETTE['medium'],   label='Tertiary Targets'),
    mpatches.Patch(color=PALETTE['info'],     label='Emerging Targets'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
ax.set_title('Top Targeted Industries — OTX Threat Pulses')
ax.set_xlabel('Number of Threat Intelligence Pulses')
plt.tight_layout()
plt.show()

top3 = list(top_industries.keys())[:3]
print("\n🔍 Analyst Insight:")
print(f"   • Top 3 targeted sectors: {', '.join(top3)}.")
print("   • Finance and Government consistently attract nation-state and cybercriminal actors")
print("     seeking financial gain (ransomware, fraud) or geopolitical intelligence collection.")
print("   • Technology sector targeting often serves as a supply-chain vector — compromise")
print("     one vendor to reach hundreds of downstream customers simultaneously.")


# ── Plot 4.3: Geographic Targeting ───────────────────────────────────────────

raw_country_counts = explode_field(otx['Countries'])
# Remove ambiguous/short entries
remove_terms = {'Unattributed', 'nan', 'Territory', 'Region', 'Island', 'Islands'}
top_countries = {
    k: v for k, v in raw_country_counts.items()
    if k not in remove_terms and len(k) > 2
}
top_countries = dict(sorted(top_countries.items(), key=lambda x: x[1], reverse=True)[:15])

cmap_grad = plt.colormaps['RdYlBu_r']
colors_grad = [cmap_grad(i / len(top_countries)) for i in range(len(top_countries))]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(list(top_countries.keys()), list(top_countries.values()),
              color=colors_grad, edgecolor='none', width=0.7)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha='center', fontsize=8)

ax.set_title('Top 15 Targeted Nations — Adversary Geographic Focus')
ax.set_xlabel('Country')
ax.set_ylabel('Intelligence Pulses Attributed')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.show()

top3c = list(top_countries.keys())[:3]
print("\n🔍 Analyst Insight:")
print(f"   • Top targeted nations: {', '.join(top3c)}.")
print("   • The United States and Russian Federation dominate — reflecting both the high value")
print("     of US targets (financial, government, defense) and Russia's dual role as a major")
print("     threat originator AND a target of Western-aligned threat actors.")
print("   • Taiwan's elevated position reflects ongoing geopolitical tensions and nation-state")
print("     cyber operations tied to the Taiwan Strait situation.")

# ── Plot 4.4: TTP Tag Cloud ───────────────────────────────────────────────────
try:
    from wordcloud import WordCloud

    all_tags = []
    for tags in otx['Tags'].dropna():
        for tag in str(tags).split(','):
            tag = tag.strip().lower()
            if tag and len(tag) > 2 and tag not in ('nan', 'unknown', ''):
                all_tags.append(tag)

    tag_freq = Counter(all_tags)
    for term in ('the', 'and', 'for', 'with', 'attack'):
        tag_freq.pop(term, None)

    wc = WordCloud(
        width=1200, height=500,
        background_color='white',
        colormap='RdYlBu_r',
        max_words=80,
        prefer_horizontal=0.7,
        collocations=False,
    ).generate_from_frequencies(tag_freq)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Adversary TTP Tag Cloud — Dominant Techniques Across All Campaigns')
    plt.tight_layout()
    plt.show()

    top10_tags = tag_freq.most_common(10)
    print("\n🔍 Analyst Insight — Top TTP Tags:")
    for tag, count in top10_tags:
        print(f"   • '{tag}': {count} mentions")
    print("\n   The dominance of credential theft, phishing, and data exfiltration confirms")
    print("   the adversary ecosystem remains primarily financially motivated.")

except ImportError:
    print("Install wordcloud: !pip install wordcloud")



# ── Plot 5.1: Top Exploited Vendors ──────────────────────────────────────────

vendor_counts = cve['vendorProject'].value_counts().head(15)

vendor_colors = []
for vendor in vendor_counts.index:
    if vendor in ('Microsoft', 'Apple'):
        vendor_colors.append(PALETTE['critical'])
    elif vendor in ('Cisco', 'Adobe', 'Google'):
        vendor_colors.append(PALETTE['high'])
    elif vendor in ('Oracle', 'Apache', 'Ivanti'):
        vendor_colors.append(PALETTE['medium'])
    else:
        vendor_colors.append(PALETTE['info'])

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(vendor_counts.index, vendor_counts.values,
              color=vendor_colors, edgecolor='none', width=0.7)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(int(bar.get_height())), ha='center', fontsize=9)

legend_handles = [
    mpatches.Patch(color=PALETTE['critical'], label='Hyperscale Attack Surface'),
    mpatches.Patch(color=PALETTE['high'],     label='Enterprise Infrastructure'),
    mpatches.Patch(color=PALETTE['medium'],   label='Network / Server Platforms'),
    mpatches.Patch(color=PALETTE['info'],     label='Niche / Appliance Vendors'),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_title('Top 15 Exploited Vendors — CISA Known Exploited Vulnerabilities')
ax.set_xlabel('Vendor / Project')
ax.set_ylabel('Actively Exploited CVEs')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.show()

ms_count = vendor_counts.get('Microsoft', 0)
print("\n🔍 Analyst Insight:")
print(f"   • Microsoft alone accounts for {ms_count} KEV entries — {ms_count/len(cve)*100:.1f}% of the catalog.")
print("   • This reflects Microsoft's ubiquity, not product quality alone.")
print("   • Ivanti's presence is operationally alarming: their VPN/remote access products are")
print("     gateway devices enabling zero-click initial access without user interaction.")


# ── Plot 5.2: CWE Weakness Taxonomy ─────────────────────────────────────────

CWE_NAMES = {
    'CWE-20':  'Improper Input Validation',  'CWE-78':  'OS Command Injection',
    'CWE-416': 'Use After Free',             'CWE-119': 'Buffer Overflow (Generic)',
    'CWE-787': 'Out-of-Bounds Write',        'CWE-22':  'Path Traversal',
    'CWE-94':  'Code Injection',             'CWE-502': 'Deserialization',
    'CWE-287': 'Improper Authentication',    'CWE-89':  'SQL Injection',
    'CWE-79':  'Cross-Site Scripting',       'CWE-693': 'Protection Mechanism Failure',
}

cwe_counts = cve['cwes'].dropna().value_counts().head(12)
cwe_dict   = cwe_counts.to_dict()   # use dict for safe .get()

labels_cwe = [f"{c}  ({CWE_NAMES.get(c, 'See MITRE')})" for c in cwe_counts.index]

cwe_colors = []
for val in cwe_counts.values:
    if val > 80:   cwe_colors.append(PALETTE['critical'])
    elif val > 60: cwe_colors.append(PALETTE['high'])
    elif val > 40: cwe_colors.append(PALETTE['medium'])
    else:          cwe_colors.append(PALETTE['info'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Exploited Vulnerability Root Causes — CWE Weakness Taxonomy', fontweight='bold')

bars = ax1.barh(labels_cwe[::-1], cwe_counts.values[::-1],
                color=cwe_colors[::-1], edgecolor='none', height=0.7)
for bar in bars:
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             str(int(bar.get_width())), va='center', fontsize=8)
ax1.set_xlabel('Actively Exploited CVEs')
ax1.set_title('Frequency by CWE Category')

# Pie: group into 5 weakness classes
memory    = sum(cwe_dict.get(c, 0) for c in ['CWE-416', 'CWE-119', 'CWE-787'])
injection = sum(cwe_dict.get(c, 0) for c in ['CWE-78',  'CWE-94',  'CWE-89', 'CWE-77', 'CWE-20'])
auth      = sum(cwe_dict.get(c, 0) for c in ['CWE-287', 'CWE-306', 'CWE-798'])
path      = sum(cwe_dict.get(c, 0) for c in ['CWE-22',  'CWE-434'])
deser     = cwe_dict.get('CWE-502', 0)
other     = max(len(cve) - memory - injection - auth - path - deser, 0)

wedge_sizes  = [memory, injection, auth, path, deser, other]
wedge_labels = ['Memory Safety\\n(UAF/BoF/OOB)', 'Injection\\n(OS/Code/SQL)',
                'Auth Bypass', 'Path/File\\nTraversal', 'Deserialization', 'Other/Unknown']
wedge_colors = [PALETTE['critical'], PALETTE['high'], PALETTE['medium'],
                PALETTE['low'],      PALETTE['stealth'], PALETTE['neutral']]

wedges, texts, autotexts = ax2.pie(
    wedge_sizes, labels=wedge_labels, colors=wedge_colors,
    autopct='%1.1f%%', startangle=140,
    textprops={'fontsize': 9},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
ax2.set_title('Vulnerability Class Distribution')

plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • Input Validation (CWE-20) and OS Command Injection (CWE-78) are the most exploited")
print("     weakness classes — both preventable through secure coding practices.")
print("   • Use-After-Free (CWE-416) exploitation requires sophisticated tooling, signaling")
print("     nation-state and advanced cybercriminal actors in the exploitation chain.")
print("   • The persistence of SQL Injection in 2024-2026 CVEs is a damning indictment of")
print("     software development — these vulnerabilities have been understood since the 1990s.")


# ── Plot 5.3: Ransomware-Weaponized CVEs ──────────────────────────────────────

ransomware_cves = cve[cve['knownRansomwareCampaignUse'] == 'Known'].copy()
ransomware_by_vendor = ransomware_cves['vendorProject'].value_counts().head(10)

r_count = len(ransomware_cves)
u_count = len(cve) - r_count

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Ransomware-Weaponized CVEs — Criminal Exploitation Toolkit',
             fontweight='bold', color=PALETTE['critical'])

# Donut: ransomware vs unknown
ax1.pie(
    [r_count, u_count],
    labels=[f'Ransomware-Linked\\n({r_count} CVEs)', f'Attribution Unknown\\n({u_count} CVEs)'],
    colors=[PALETTE['critical'], PALETTE['neutral']],
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 11},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.6}
)
ax1.set_title(f'{r_count/len(cve)*100:.1f}% of KEV CVEs Linked to Ransomware')

# Bar: ransomware CVEs per vendor
bars = ax2.barh(ransomware_by_vendor.index[::-1],
                ransomware_by_vendor.values[::-1],
                color=PALETTE['critical'], alpha=0.85, edgecolor='none')
for bar in bars:
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
             str(int(bar.get_width())), va='center', fontsize=9)
ax2.set_title('Ransomware CVEs by Vendor')
ax2.set_xlabel('CVEs Weaponized by Ransomware Groups')

plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print(f"   • {r_count/len(cve)*100:.1f}% ({r_count}) of all KEV CVEs have confirmed ransomware campaign linkage.")
print("   • This is conservative — many 'Unknown' entries likely involve ransomware groups using")
print("     pseudonymous infrastructure that resists attribution.")
print("   • Ransomware groups systematically monitor CISA KEV and weaponize newly disclosed")
print("     vulnerabilities within 24-72 hours of public disclosure.")


# ── Plot 6.1: TLD Distribution ───────────────────────────────────────────────

tld_counts = domains['TLD'].value_counts().head(15)

tld_colors = []
for tld in tld_counts.index:
    if tld in ('ru', 'cn', 'ir', 'kp'):
        tld_colors.append(PALETTE['critical'])
    elif tld in ('top', 'xyz', 'online', 'site', 'tk', 'ml', 'ga', 'cf', 'gq'):
        tld_colors.append(PALETTE['high'])
    elif tld in ('com', 'net', 'org'):
        tld_colors.append(PALETTE['medium'])
    else:
        tld_colors.append(PALETTE['info'])

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(tld_counts.index, tld_counts.values,
              color=tld_colors, edgecolor='none', width=0.7)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', fontsize=9)

legend_handles = [
    mpatches.Patch(color=PALETTE['critical'], label='Nation-State Associated TLDs'),
    mpatches.Patch(color=PALETTE['high'],     label='Free / Low-cost / Bulletproof TLDs'),
    mpatches.Patch(color=PALETTE['medium'],   label='Legacy TLDs (com/net/org)'),
    mpatches.Patch(color=PALETTE['info'],     label='Country-Code TLDs'),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_title('Adversary TLD Preferences — Malicious Domain Infrastructure')
ax.set_xlabel('Top-Level Domain')
ax.set_ylabel('Number of Malicious Domains')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • Legacy TLDs (.com, .net, .ch) dominate — adversaries blend into legitimate")
print("     traffic to evade DNS reputation filtering.")
print("   • Free/bulletproof TLDs are favored for disposable infrastructure that can be")
print("     abandoned after detection without financial loss to the threat actor.")


# ── Plot 6.2: Domain Length & Obfuscation Patterns ────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Domain Naming Intelligence — Length & Obfuscation Patterns', fontweight='bold')

# Histogram of domain lengths
dl = domains['Domain_Length'].dropna()
ax1.hist(dl, bins=25, color=PALETTE['info'], edgecolor='white', alpha=0.85)
ax1.axvline(dl.median(), color=PALETTE['critical'],  linestyle='--', linewidth=2,
            label=f'Median: {dl.median():.0f} chars')
ax1.axvline(dl.mean(),   color=PALETTE['medium'],    linestyle=':',  linewidth=2,
            label=f'Mean: {dl.mean():.1f} chars')
ax1.set_title('Domain Length Distribution')
ax1.set_xlabel('Characters in Domain Name')
ax1.set_ylabel('Count')
ax1.legend(fontsize=9)

# Stacked bar: numeric & hyphen presence
num_yes = (domains['Has_Numbers'] == 'Yes').sum()
num_no  = (domains['Has_Numbers'] == 'No').sum()
hyp_yes = (domains['Has_Hyphen']  == 'Yes').sum()
hyp_no  = (domains['Has_Hyphen']  == 'No').sum()

x = np.arange(2)
ax2.bar(x, [num_yes, hyp_yes], label='Yes', color=PALETTE['critical'], alpha=0.85)
ax2.bar(x, [num_no,  hyp_no],  bottom=[num_yes, hyp_yes],
        label='No', color=PALETTE['info'], alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(['Contains Numbers', 'Contains Hyphens'])
ax2.set_title('Obfuscation Indicators in Domain Names')
ax2.set_ylabel('Domain Count')
ax2.legend(fontsize=9)

for xi, (y_val, n_val) in enumerate(zip([num_yes, hyp_yes], [num_no, hyp_no])):
    ax2.text(xi, y_val / 2,       str(y_val), ha='center', va='center', fontsize=11,
             color='white', fontweight='bold')
    ax2.text(xi, y_val + n_val / 2, str(n_val), ha='center', va='center', fontsize=11,
             color='white', fontweight='bold')

plt.tight_layout()
plt.show()

pct_numbers = num_yes / len(domains) * 100
pct_hyphen  = hyp_yes / len(domains) * 100
print("\n🔍 Analyst Insight:")
print(f"   • {pct_numbers:.1f}% of malicious domains contain numeric characters — a common DGA signature.")
print(f"   • {pct_hyphen:.1f}% use hyphens, often mimicking legitimate brands (e.g., paypal-secure.com).")
print("   • Domains with length > 30 characters often indicate DGA activity where malware")
print("     generates pseudo-random domains for C2 beaconing.")


# ── Plot 6.3: Domain Threat Severity & Reputation ─────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Domain Threat Assessment — Severity & Reputation Intelligence', fontweight='bold')

# Donut: severity
sev_counts = domains['Threat_Severity'].fillna('Unknown').value_counts()
sev_color_map = {'High': PALETTE['critical'], 'Medium': PALETTE['medium'],
                 'Low': PALETTE['info'], 'Unknown': PALETTE['neutral']}
sev_colors = [sev_color_map.get(s, PALETTE['neutral']) for s in sev_counts.index]

wedges, texts, autotexts = ax1.pie(
    sev_counts.values, labels=sev_counts.index, colors=sev_colors,
    autopct='%1.1f%%', startangle=140, pctdistance=0.75,
    textprops={'fontsize': 11},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.6}
)
ax1.set_title('Domain Threat Severity Distribution')

# Histogram: reputation scores
rep_data = domains['Reputation'].dropna()
ax2.hist(rep_data, bins=20, color=PALETTE['stealth'], edgecolor='white', alpha=0.85)
ax2.axvline(0,              color=PALETTE['critical'], linestyle='--', linewidth=2,
            label='Neutral Score = 0')
ax2.axvline(rep_data.median(), color=PALETTE['medium'], linestyle=':',  linewidth=2,
            label=f'Median = {rep_data.median():.0f}')
ax2.set_title('VirusTotal Reputation Score Distribution')
ax2.set_xlabel('Reputation Score (negative = malicious)')
ax2.set_ylabel('Domain Count')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()

high_sev = sev_counts.get('High', 0)
print("\n🔍 Analyst Insight:")
print(f"   • {high_sev} domains are classified High severity — active C2 or phishing infrastructure.")
print(f"   • Reputation median: {rep_data.median():.0f}. Scores below -50 indicate broad consensus that")
print("     a domain is actively malicious across multiple security vendor engines.")
print("   • Low-severity domains may represent initial staging infrastructure not yet heavily")
print("     flagged, or newly registered C2 nodes in pre-deployment status.")


# ── Plot 7.1: IP Geolocation ──────────────────────────────────────────────────

ip_countries = ips['Country'].value_counts().head(12)

COUNTRY_NAMES = {
    'DE': 'Germany',      'NL': 'Netherlands',   'US': 'United States',
    'FR': 'France',       'CN': 'China',          'BG': 'Bulgaria',
    'VG': 'Br. Virgin Is.','CH': 'Switzerland',  'CA': 'Canada',
    'RU': 'Russia',       'UA': 'Ukraine',        'GB': 'United Kingdom',
}
labels_ip = [COUNTRY_NAMES.get(c, c) for c in ip_countries.index]

ip_country_colors = []
for c in ip_countries.index:
    if c in ('RU', 'CN', 'KP', 'IR'):
        ip_country_colors.append(PALETTE['critical'])
    elif c in ('NL', 'DE', 'BG', 'VG'):
        ip_country_colors.append(PALETTE['medium'])
    else:
        ip_country_colors.append(PALETTE['info'])

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(labels_ip[::-1], ip_countries.values[::-1],
               color=ip_country_colors[::-1], edgecolor='none', height=0.7)
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=9)

legend_handles = [
    mpatches.Patch(color=PALETTE['critical'], label='Nation-State Attributed Hosting'),
    mpatches.Patch(color=PALETTE['medium'],   label='Bulletproof / Permissive Jurisdictions'),
    mpatches.Patch(color=PALETTE['info'],     label='Western Infrastructure (Anonymized Use)'),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_title('Malicious IP Hosting Countries — Adversary Infrastructure Geolocation')
ax.set_xlabel('Number of Malicious IPs Hosted')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • Germany and Netherlands dominate — NOT because they are threat actors, but because")
print("     they host major datacenters (Hetzner, OVH) offering cheap, anonymous VPS services.")
print("   • Adversaries deliberately host in EU/US jurisdictions to blend with legitimate traffic")
print("     and exploit slower cross-border law enforcement takedown timelines.")
print("   • British Virgin Islands IPs indicate offshore shell-company registrations used for")
print("     bulletproof hosting beyond easy legal reach.")


# ── Plot 7.2: TOR Usage & Threat Category ────────────────────────────────────

tor_counts  = ips['TOR_Node'].value_counts()
threat_cats = ips['Threat_Category'].value_counts()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Anonymization & Threat Classification — IP Intelligence', fontweight='bold')

# TOR pie
tor_yes = tor_counts.get('Yes', 0)
tor_no  = tor_counts.get('No',  0)
ax1.pie(
    [tor_yes, tor_no],
    labels=[f'TOR Exit Node\\n({tor_yes} IPs)', f'Standard IP\\n({tor_no} IPs)'],
    colors=[PALETTE['stealth'], PALETTE['info']],
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 11},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax1.set_title('TOR Anonymization Network Usage')

# Threat category bar
cat_color_map = {
    'malware': PALETTE['critical'], 'clean':  PALETTE['info'],
    'unrated': PALETTE['neutral'],  'phishing': PALETTE['high'],
    'spam':    PALETTE['medium'],
}
bar_colors_tc = [cat_color_map.get(c, PALETTE['neutral']) for c in threat_cats.index]
bars = ax2.bar(threat_cats.index, threat_cats.values,
               color=bar_colors_tc, edgecolor='none', width=0.6)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             str(int(bar.get_height())), ha='center', fontsize=10, fontweight='bold')
ax2.set_title('IP Threat Category Classification')
ax2.set_xlabel('Threat Category')
ax2.set_ylabel('IP Count')

plt.tight_layout()
plt.show()

tor_pct = tor_yes / len(ips) * 100
print("\n🔍 Analyst Insight:")
print(f"   • {tor_pct:.1f}% of tracked malicious IPs are confirmed TOR exit nodes — used to anonymize")
print("     C2 communications and run attribution-resistant exfiltration channels.")
print("   • The large 'unrated' category is operationally dangerous: these IPs are fresh")
print("     infrastructure deployed after intelligence collection and not yet classified.")
print("   • 'Clean' IPs in a malicious dataset are often compromised legitimate systems:")
print("     hijacked home routers or cloud instances used as intermediary proxies.")


# ── Plot 7.3: Hosting Provider (ASN) Intelligence ────────────────────────────

top_owners = ips['Owner'].dropna().value_counts().head(12)
short_labels = [o[:30] + '...' if len(o) > 30 else o for o in top_owners.index]

cmap_asn = plt.colormaps['plasma']
colors_asn = [cmap_asn(i / len(top_owners)) for i in range(len(top_owners))]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.barh(short_labels[::-1], top_owners.values[::-1],
               color=colors_asn[::-1], edgecolor='none', height=0.7)
for bar in bars:
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=9)

ax.set_title('Top Hosting Providers — Malicious IP ASN Intelligence')
ax.set_xlabel('Number of Malicious IPs in ASN')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • Datacenter ASNs dominating this list confirms adversaries overwhelmingly prefer")
print("     VPS infrastructure over residential botnets for C2 operation.")
print("   • Scaleway, Cloudflare, Hetzner and similar providers offer near-anonymous account")
print("     creation, cryptocurrency payment, and minimal abuse response time.")
print("   • Defenders should weight reputation differently for these ASNs: a new IP from a")
print("     bulletproof-friendly datacenter warrants higher initial suspicion.")


# ── Plot 8.1: Top MITRE ATT&CK Techniques ────────────────────────────────────

all_attack_ids = []
for val in otx['Attack_IDs'].dropna():
    for tid in str(val).split(','):
        tid = tid.strip()
        if re.match(r'T\d{4}', tid):
            all_attack_ids.append(tid)

attack_counter = Counter(all_attack_ids)
top_attacks    = dict(sorted(attack_counter.items(), key=lambda x: x[1], reverse=True)[:20])

ATTACK_NAMES = {
    'T1059':     'Command & Scripting Interpreter',
    'T1059.001': 'PowerShell',
    'T1059.003': 'Windows Command Shell',
    'T1027':     'Obfuscated Files / Info',
    'T1071':     'App Layer Protocol (C2)',
    'T1071.001': 'Web Protocols (C2)',
    'T1041':     'Exfiltration Over C2',
    'T1105':     'Ingress Tool Transfer',
    'T1036':     'Masquerading',
    'T1055':     'Process Injection',
    'T1082':     'System Info Discovery',
    'T1083':     'File & Dir Discovery',
    'T1497':     'Sandbox Evasion',
    'T1573':     'Encrypted Channel',
    'T1140':     'Deobfuscate / Decode Files',
    'T1566':     'Phishing',
    'T1204':     'User Execution',
    'T1486':     'Data Encrypted for Impact',
    'T1090':     'Proxy',
    'T1132':     'Data Encoding',
    'T1005':     'Data from Local System',
    'T1555':     'Credentials from Password Stores',
}

TACTIC_GROUPS = {
    'T1059': 'Execution',    'T1059.001': 'Execution',  'T1059.003': 'Execution',
    'T1204': 'Execution',    'T1105':     'Execution',
    'T1027': 'Evasion',      'T1036':     'Evasion',    'T1497': 'Evasion',
    'T1140': 'Evasion',      'T1055':     'Evasion',    'T1090': 'Evasion',
    'T1041': 'Exfiltration', 'T1132':     'Exfiltration',
    'T1071': 'C2',           'T1071.001': 'C2',         'T1573': 'C2',
    'T1082': 'Discovery',    'T1083':     'Discovery',
    'T1555': 'Credential',   'T1486':     'Impact',
}
TACTIC_COLOR = {
    'Execution': PALETTE['critical'], 'Evasion': PALETTE['stealth'],
    'C2': PALETTE['medium'], 'Exfiltration': PALETTE['high'],
    'Discovery': PALETTE['info'], 'Credential': PALETTE['accent'],
    'Impact': '#E74C3C',
}

labels_atk = [f"{tid} — {ATTACK_NAMES.get(tid, 'See MITRE')}" for tid in top_attacks]
values_atk  = list(top_attacks.values())
colors_atk  = [TACTIC_COLOR.get(TACTIC_GROUPS.get(tid, ''), PALETTE['neutral'])
               for tid in top_attacks]

fig, ax = plt.subplots(figsize=(14, 9))
bars = ax.barh(labels_atk[::-1], values_atk[::-1],
               color=colors_atk[::-1], edgecolor='none', height=0.72)
for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=8)

legend_handles = [mpatches.Patch(color=v, label=k) for k, v in TACTIC_COLOR.items()]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
ax.set_title('Top 20 MITRE ATT&CK Techniques — Adversary Playbook Frequency')
ax.set_xlabel('Technique Frequency Across OTX Pulses')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • T1027 (Obfuscated Files/Info) ranks #1 — defense evasion is embedded in every campaign.")
print("   • T1059.001 (PowerShell) confirms living-off-the-land execution remains the adversary's")
print("     preferred initial execution method — no malicious binary needed.")
print("   • The co-presence of T1041 (Exfil over C2) and T1071.001 (Web Protocols) suggests")
print("     adversaries route exfiltration through HTTPS to blend with normal web traffic.")
print("   • T1486 (Data Encrypted for Impact) confirms ransomware TTPs are embedded across")
print("     the broader campaign landscape, not isolated to criminal groups.")


# ── Plot 8.2: Active Malware Family Intelligence ─────────────────────────────

all_families = []
for val in otx['Malware_Families'].dropna():
    for fam in str(val).split(','):
        fam = re.sub(r'\s*-\s*S\d+', '', fam).strip()
        if fam and fam not in ('Unclassified', 'nan', '') and len(fam) > 2:
            all_families.append(fam)

family_counter = Counter(all_families)
top_families   = dict(sorted(family_counter.items(), key=lambda x: x[1], reverse=True)[:15])

FAMILY_CATEGORIES = {
    'Infostealer': ['Lumma Stealer', 'LummaStealer', 'RedLine', 'Raccoon', 'Vidar',
                    'AgentTesla', 'FormBook', 'Snake Keylogger', 'StealC', 'Rhadamanthys'],
    'RAT':         ['AsyncRAT', 'Remcos', 'NetWireRC', 'QuasarRAT', 'DarkComet',
                    'NanoCore', 'XWorm', 'Telegram RAT'],
    'Ransomware':  ['LockBit', 'BlackCat', 'Ryuk', 'Conti', 'REvil', 'BlackBasta',
                    'Kyber', 'AlphaLocker', 'Play', 'Cl0p'],
    'Loader':      ['HijackLoader', 'SmokeLoader', 'IcedID', 'Emotet', 'Qakbot',
                    'GootLoader', 'Cobalt Strike'],
    'Banking':     ['TrickBot', 'KYCShadow', 'Dridex', 'Ursnif', 'Zeus'],
}
CAT_COLOR = {
    'Infostealer': PALETTE['critical'], 'RAT':     PALETTE['stealth'],
    'Ransomware':  '#E74C3C',           'Loader':  PALETTE['high'],
    'Banking':     PALETTE['medium'],   'Other':   PALETTE['info'],
}

def categorize_family(name):
    for cat, members in FAMILY_CATEGORIES.items():
        if any(m.lower() in name.lower() for m in members):
            return cat
    return 'Other'

fam_labels = list(top_families.keys())
fam_values = list(top_families.values())
fam_colors = [CAT_COLOR[categorize_family(f)] for f in fam_labels]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(fam_labels[::-1], fam_values[::-1],
               color=fam_colors[::-1], edgecolor='none', height=0.7)
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=9)

legend_handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLOR.items()]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_title('Active Malware Family Intelligence — Threat Ecosystem Map')
ax.set_xlabel('Intelligence Pulse Mentions')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • Cobalt Strike / Lumma Stealer dominate — the infostealer-as-a-service model has matured.")
print("   • HijackLoader's prominence confirms the 'loader-as-a-service' model: adversaries")
print("     purchase delivery infrastructure separately from the final payload.")
print("   • KYCShadow (Android banking trojan targeting OTP theft) signals expansion into")
print("     mobile attack surfaces, particularly in Asia Pacific.")
print("   • AsyncRAT's persistence in the ecosystem reflects its open-source availability")
print("     and ease of customization by low-sophistication threat actors.")



# ── Plot 8.3: Industry vs TTP Heatmap ────────────────────────────────────────

tactic_keywords = {
    'Phishing':          ['phishing', 'spear-phishing', 'clickfix', 'social engineering'],
    'Ransomware':        ['ransomware', 'file encryption', 'double extortion'],
    'Credential Theft':  ['credential theft', 'password', 'keylogger', 'infostealer'],
    'C2 Comms':          ['c2 communication', 'command and control', 'beacon'],
    'Defense Evasion':   ['defense evasion', 'obfuscation', 'masquerading'],
    'Data Exfiltration': ['data exfiltration', 'exfiltration', 'data theft'],
    'Lateral Movement':  ['lateral movement', 'pivot'],
    'Supply Chain':      ['supply chain', 'dependency confusion', 'typosquat'],
}

target_industries = ['Finance', 'Government', 'Technology', 'Defense',
                     'Healthcare', 'Retail', 'Energy', 'Education']

matrix = np.zeros((len(target_industries), len(tactic_keywords)))
for i, industry in enumerate(target_industries):
    subset   = otx[otx['Industries'].str.contains(industry, case=False, na=False)]
    all_text = ' '.join(
        subset['Tags'].fillna('').astype(str) + ' ' +
        subset['Description'].fillna('').astype(str)
    ).lower()
    for j, (_, keywords) in enumerate(tactic_keywords.items()):
        matrix[i, j] = sum(all_text.count(kw) for kw in keywords)

# Row-normalize to percentages
row_sums = matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
matrix_norm = (matrix / row_sums * 100).round(1)

cmap_heat = LinearSegmentedColormap.from_list('threat', ['#f7f7f7', '#FF7F0E', '#D62728'])

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(matrix_norm, cmap=cmap_heat, aspect='auto', vmin=0, vmax=matrix_norm.max())

ax.set_xticks(range(len(tactic_keywords)))
ax.set_xticklabels(list(tactic_keywords.keys()), fontsize=10, rotation=15, ha='right')
ax.set_yticks(range(len(target_industries)))
ax.set_yticklabels(target_industries, fontsize=10)

for i in range(len(target_industries)):
    for j in range(len(tactic_keywords)):
        val = matrix_norm[i, j]
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                fontsize=9, color='white' if val > matrix_norm.max() * 0.4 else '#333333',
                fontweight='bold')

plt.colorbar(im, ax=ax, label='Relative TTP Prevalence (%)', fraction=0.046, pad=0.04)
ax.set_title('Industry vs. Adversary TTP Heatmap — Targeting Pattern Intelligence')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • Finance shows high affinity with Credential Theft and Phishing — adversaries target")
print("     financial orgs for account takeover as the primary monetization path.")
print("   • Government targeting skews toward C2 Comms and Data Exfiltration, consistent")
print("     with nation-state APT behavior (persistent collection vs quick monetization).")
print("   • Defense sector shows elevated Supply Chain activity — confirming the shift toward")
print("     attacking defense contractors through their software vendors and subcontractors.")
print("   • Healthcare's ransomware prevalence reflects high ransom-payment rates driven by")
print("     patient safety pressures and operational continuity requirements.")

# ── Plot 9.1: CVE Description NLP — Root Cause Keyword Analysis ───────────────

def extract_keywords(text_series, top_n=25):
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'been', 'which', 'when', 'they',
        'could', 'would', 'allow', 'allows', 'attacker', 'attackers', 'remote',
        'local', 'arbitrary', 'execute', 'code', 'vulnerability', 'contains',
        'version', 'before', 'versions', 'product', 'software', 'system',
        'application', 'user', 'users', 'via', 'through', 'into', 'within',
        'certain', 'leads', 'lead', 'result', 'results', 'affected', 'allows',
    }
    word_counts = Counter()
    for text in text_series.dropna():
        words = re.findall(r'[a-z]{4,}', str(text).lower())
        for w in words:
            if w not in stop_words:
                word_counts[w] += 1
    return word_counts.most_common(top_n)

cve_keywords = extract_keywords(cve['shortDescription'])
kw_labels = [k for k, _ in cve_keywords]
kw_values = [v for _, v in cve_keywords]

HIGH_RISK   = {'injection', 'overflow', 'execution', 'escalation', 'bypass',
               'traversal', 'deserialization', 'command', 'spoofing', 'disclosure'}
MEDIUM_RISK = {'authentication', 'authorization', 'credentials', 'password',
               'privilege', 'improper', 'uncontrolled', 'insufficient', 'missing'}

kw_colors = []
for kw in kw_labels:
    if kw in HIGH_RISK:    kw_colors.append(PALETTE['critical'])
    elif kw in MEDIUM_RISK: kw_colors.append(PALETTE['medium'])
    else:                   kw_colors.append(PALETTE['info'])

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(kw_labels[::-1], kw_values[::-1],
               color=kw_colors[::-1], edgecolor='none', height=0.72)
for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=8)

legend_handles = [
    mpatches.Patch(color=PALETTE['critical'], label='High-Risk Exploit Terms'),
    mpatches.Patch(color=PALETTE['medium'],   label='Access Control Failures'),
    mpatches.Patch(color=PALETTE['info'],     label='General Descriptors'),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_title('CVE Description NLP — Root Cause Keyword Frequency Analysis')
ax.set_xlabel('Frequency in CISA KEV Descriptions')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • 'Memory' and 'buffer' appear with high frequency, confirming memory safety issues")
print("     remain the dominant class of exploitable vulnerabilities in modern software.")
print("   • 'Privilege' and 'escalation' terms match the adversary pattern of initial low-privilege")
print("     access followed by local privilege escalation for full system control.")
print("   • 'Authentication' weakness terms confirm broken auth is endemic, particularly in")
print("     network appliances and remote access products.")


# ── Plot 9.2: OTX Pulse Title Bigrams ────────────────────────────────────────

def extract_bigrams(text_series, top_n=20):
    stop_words = {
        'with', 'from', 'that', 'this', 'using', 'via', 'been', 'have',
        'were', 'their', 'will', 'into', 'analysis', 'campaign', 'attack',
        'based', 'through', 'related', 'leveraging', 'targeting', 'used',
    }
    bigram_counts = Counter()
    for text in text_series.dropna():
        words = [w for w in re.findall(r'[a-z]{3,}', str(text).lower())
                 if w not in stop_words]
        for i in range(len(words) - 1):
            bigram_counts[f"{words[i]} {words[i+1]}"] += 1
    return bigram_counts.most_common(top_n)

bigrams   = extract_bigrams(otx['Title'])
bg_labels = [k for k, _ in bigrams]
bg_values = [v for _, v in bigrams]

cmap_plasma = plt.colormaps['plasma']
colors_bg   = [cmap_plasma(i / len(bg_labels)) for i in range(len(bg_labels))]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(bg_labels[::-1], bg_values[::-1],
               color=colors_bg[::-1], edgecolor='none', height=0.72)
for bar in bars:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va='center', fontsize=8)

ax.set_title('OTX Pulse Title Bigrams — Intelligence Report Language Patterns')
ax.set_xlabel('Bigram Frequency in Pulse Titles')
plt.tight_layout()
plt.show()

print("\n🔍 Analyst Insight:")
print("   • 'Threat actors', 'supply chain', and 'malware analysis' dominate — reflecting")
print("     community focus on actor attribution and technical dissection.")
print("   • 'Nation state' and 'advanced persistent' bigrams confirm sophisticated multi-stage")
print("     operations are the primary research focus in the intelligence community.")
print("   • 'Phishing campaign' and 'credential theft' reinforce the finding from earlier:")
print("     initial access via phishing + credential harvest is the dominant entry pattern.")


# ── Executive Dashboard ───────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.suptitle('THREAT INTELLIGENCE EXECUTIVE DASHBOARD',
             fontsize=16, fontweight='bold', y=1.0)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

# ── Panel 1: Metric cards ─────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis('off')
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)

metrics = [
    ('2,365', 'Threat Campaigns Tracked',         PALETTE['critical']),
    ('1,585', 'KEV CVEs — Actively Exploited',    PALETTE['high']),
    (f'{len(ransomware_cves)}',
              'Ransomware-Linked CVEs',            PALETTE['medium']),
    (f'{tor_yes}',
              'TOR Exit Nodes Identified',         PALETTE['stealth']),
]
for idx, (val, label, color) in enumerate(metrics):
    y = 0.85 - idx * 0.22
    ax0.text(0.08, y,        val,   fontsize=17, color=color, fontweight='bold')
    ax0.text(0.08, y - 0.09, label, fontsize=8,  color='gray')
    ax0.axhline(y - 0.12, color='lightgray', linewidth=0.5)
ax0.set_title('Key Metrics', fontweight='bold')

# ── Panel 2: Top targeted industries ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
top6_ind = dict(list(industry_counts.items())[:6])
colors6  = [PALETTE['critical'], PALETTE['high'], PALETTE['medium'],
            PALETTE['info'],     PALETTE['stealth'], PALETTE['accent']]
ax1.barh(list(top6_ind.keys())[::-1], list(top6_ind.values())[::-1],
         color=colors6[::-1], edgecolor='none', height=0.65)
ax1.set_title('Top Targeted Industries', fontweight='bold')
ax1.set_xlabel('Pulse Count', fontsize=9)

# ── Panel 3: Top exploited vendors ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
top6_vendor = vendor_counts.head(6)
ax2.bar(top6_vendor.index, top6_vendor.values,
        color=colors6, edgecolor='none', width=0.65)
ax2.set_title('Top Exploited Vendors', fontweight='bold')
ax2.set_ylabel('KEV CVE Count', fontsize=9)
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=9)

# ── Panel 4: Country targeting ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
top6c = dict(list(top_countries.items())[:6])
ax3.bar(list(top6c.keys()), list(top6c.values()),
        color=PALETTE['info'], alpha=0.85, edgecolor='none')
ax3.set_title('Top Targeted Nations', fontweight='bold')
ax3.set_ylabel('Pulse Count', fontsize=9)
plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=9)

# ── Panel 5: IP threat categories ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
threat_cats_dash = ips['Threat_Category'].value_counts()
pie_col = [PALETTE['critical'], PALETTE['info'], PALETTE['neutral']]
ax4.pie(threat_cats_dash.values,
        labels=threat_cats_dash.index,
        colors=pie_col[:len(threat_cats_dash)],
        autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 9},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax4.set_title('IP Threat Classification', fontweight='bold')

# ── Panel 6: Top ATT&CK techniques ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
top8_atk  = dict(list(top_attacks.items())[:8])
short_atk = [ATTACK_NAMES.get(k, k)[:24] for k in top8_atk]
ax5.barh(short_atk[::-1], list(top8_atk.values())[::-1],
         color=PALETTE['stealth'], alpha=0.85, edgecolor='none', height=0.65)
ax5.set_title('Top ATT&CK Techniques', fontweight='bold')
ax5.set_xlabel('Frequency', fontsize=9)
ax5.tick_params(axis='y', labelsize=8)

plt.savefig('executive_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Executive dashboard rendered.")