import requests
from bs4 import BeautifulSoup
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

base_url = "https://www.shl.com"
headers = {"User-Agent": "Mozilla/5.0"}
data = []


# Function to scrape individual product detail page
# Function to scrape individual product detail page
def scrape_product(product_info):
    title, relative_link, remote_testing, adaptive_rt, test_type = product_info
    product_url = base_url + relative_link
    try:
        response = requests.get(product_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"✗ Failed to fetch product page: {product_url}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        duration = "N/A"
        description = "N/A"
        job_levels = "N/A"

        # Look for product detail rows
        rows = soup.find_all("div", class_="product-catalogue-training-calendar__row")
        for row in rows:
            h4 = row.find("h4")
            if not h4:
                continue

            label = h4.text.strip()
            content = row.find("p").text.strip() if row.find("p") else ""

            if "Assessment length" in label:
                match = re.search(r"(\d+)", content)
                if match:
                    duration = int(match.group(1))
            elif "Description" in label:
                description = content
            elif "Job levels" in label:
                job_levels = content

        print(f"✓ Scraped: {title}")
        return {
            "title": title,
            "link": product_url,
            "remote_testing": remote_testing,
            "adaptive_rt": adaptive_rt,
            "test_type": test_type,
            "duration": duration,
            "description": description,
            "job_levels": job_levels,
        }

    except Exception as e:
        print(f"✗ Error scraping {product_url}: {e}")
        return None


# Main loop through paginated catalog
for page in range(0, 31):  # Adjust number of pages as needed
    url = f"{base_url}/solutions/products/product-catalog/?start={page*12}&type=1"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"✗ Failed to fetch catalog page {page + 1}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")
    rows = soup.find_all("tr", attrs={"data-entity-id": True})
    product_list = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        a_tag = cols[0].find("a")
        if not a_tag:
            continue

        title = a_tag.text.strip()
        relative_link = a_tag.get("href")

        remote_testing = (
            "Yes" if cols[1].find("span", class_="catalogue__circle -yes") else "No"
        )
        adaptive_rt = (
            "Yes" if cols[2].find("span", class_="catalogue__circle -yes") else "No"
        )

        test_type_span = cols[3].find("span", class_="product-catalogue__key")
        test_type = test_type_span.text.strip() if test_type_span else "N/A"

        product_list.append(
            (title, relative_link, remote_testing, adaptive_rt, test_type)
        )

    print(f"🧵 Page {page + 1}: Starting {len(product_list)} threads...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_product = {
            executor.submit(scrape_product, item): item for item in product_list
        }
        for future in as_completed(future_to_product):
            result = future.result()
            if result:
                data.append(result)

    time.sleep(1)

# Save final data to JSON
with open("data_scraped.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ All done! Data saved to data_scraped.json")
