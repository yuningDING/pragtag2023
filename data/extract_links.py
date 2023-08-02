import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import colorama
import os


colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW

# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()

total_urls_visited = 0

domain_name = 'https://f1000research.com'

target_folder = 'gateways'
if not os.path.exists(target_folder):
    os.mkdir(target_folder)


def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_website_links(url):
    """
    Returns all URLs that is found on `url` in which it belongs to the same website
    """
    # all URLs of `url`
    urls = set()
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue
        # join the URL if it's relative (not absolute link)
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        # remove URL GET parameters, URL fragments, etc.
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            # not a valid URL
            continue
        if href in internal_urls:
            # already in the set
            continue
        if domain_name not in href:
            # external link
            if href not in external_urls:
                print(f"{GRAY}[!] External link: {href}{RESET}")
                external_urls.add(href)
            continue
        print(f"{GREEN}[*] Internal link: {href}{RESET}")
        urls.add(href)
        internal_urls.add(href)
    return urls


def crawl(url, max_urls=30):

    global total_urls_visited
    total_urls_visited += 1
    print(f"{YELLOW}[*] Crawling: {url}{RESET}")
    links = get_all_website_links(url)


def collect_gateway(url_list, gateway):

    for url in url_list:
        domain_name = urlparse(url).netloc
        crawl(url, max_urls=500)

    print("[+] Total Internal links:", len(internal_urls))
    print("[+] Total External links:", len(external_urls))

    # save the internal links to a file
    with open(os.path.join(target_folder, f"{gateway}_internal_links.txt"), "w") as f:
        for internal_link in internal_urls:
            if '/articles/' in str(internal_link):
                print(internal_link.strip(), file=f)


# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()
total_urls_visited = 0
domain = 'rpkg'
rpack_1 = 'https://f1000research.com/gateways/rpackage?&show=119&page=1'
rpack_2 = 'https://f1000research.com/gateways/rpackage?&show=100&page=2'
collect_gateway(url_list=[rpack_1, rpack_2], gateway=domain)

# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()
total_urls_visited = 0
domain = 'scip'
scip_1 = 'https://f1000research.com/gateways/research_on_research?&show=100&page=1'
scip_2 = 'https://f1000research.com/gateways/research_on_research?&show=100&page=2'
scip_3 = 'https://f1000research.com/gateways/research_on_research?&show=100&page=3'
collect_gateway(url_list=[scip_1, scip_2, scip_3], gateway=domain)

# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()
total_urls_visited = 0
domain = 'iscb'
iscb_1 = 'https://f1000research.com/gateways/bioinformaticsgw?&show=100&page=1'
iscb_2 = 'https://f1000research.com/gateways/bioinformaticsgw?&show=100&page=2'
iscb_3 = 'https://f1000research.com/gateways/bioinformaticsgw?&show=100&page=3'
iscb_4 = 'https://f1000research.com/gateways/bioinformaticsgw?&show=100&page=4'
collect_gateway(url_list=[iscb_1, iscb_2, iscb_3, iscb_4], gateway=domain)

# initialize the set of links (unique links)
internal_urls = set()
external_urls = set()
total_urls_visited = 0
domain = 'diso'
diso_1 = 'https://f1000research.com/gateways/disease_outbreaks?&show=100&page=1'
diso_2 = 'https://f1000research.com/gateways/disease_outbreaks?&show=100&page=2'
diso_3 = 'https://f1000research.com/gateways/disease_outbreaks?&show=100&page=3'
diso_4 = 'https://f1000research.com/gateways/disease_outbreaks?&show=100&page=4'
diso_5 = 'https://f1000research.com/gateways/disease_outbreaks?&show=100&page=5'
collect_gateway(url_list=[diso_1, diso_2, diso_3, diso_4, diso_5], gateway=domain)