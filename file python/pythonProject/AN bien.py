from playwright.sync_api import sync_playwright
import pandas as pd


def main():
    with sync_playwright() as p:
        # Dates for the search
        checkin_date = '2024-06-15'
        checkout_date = '2024-06-16'

        page_url = f'https://www.booking.com/searchresults.vi.html?ss=Vu%CC%83ng+Ta%CC%80u&ssne=Vu%CC%83ng+Ta%CC%80u&ssne_untouched=Vu%CC%83ng+Ta%CC%80u&efdco=1&label=60735e40280611ef9e7b3bc003d56c89&sid=16598d31e55bae96340b614b4219413a&aid=2167732&lang=vi&sb=1&src_elem=sb&src=searchresults&dest_id=-3733750&dest_type=city&checkin=2024-06-15&checkout=2024-06-16&group_adults=2&no_rooms=1&group_children=0&sb_travel_purpose=leisure'

        # Launch Microsoft Edge using the Chromium API
        browser = p.chromium.launch(channel="msedge", headless=False)
        page = browser.new_page()
        page.goto(page_url, timeout=60000)

        hotels = page.locator('//div[@data-testid="property-card"]').all()
        print(f'There are: {len(hotels)} hotels.')

        hotels_list = []
        for hotel in hotels:
            hotel_dict = {}
            try:
                hotel_dict['hotel'] = hotel.locator('//div[@data-testid="title"]').inner_text().strip()
            except:
                hotel_dict['hotel'] = 'N/A'
            try:
                hotel_dict['price'] = hotel.locator('//span[@data-testid="price-and-discounted-price"]').inner_text().strip()
            except:
                hotel_dict['price'] = 'N/A'
            try:
                hotel_dict['score'] = hotel.locator('//div[@data-testid="review-score"]/div[1]').inner_text().strip()
            except:
                hotel_dict['score'] = 'N/A'
            try:
                hotel_dict['avg review'] = hotel.locator('//div[@data-testid="review-score"]/div[2]/div[1]').inner_text().strip()
            except:
                hotel_dict['avg review'] = 'N/A'
            try:
                hotel_dict['reviews count'] = hotel.locator('//div[@data-testid="review-score"]/div[2]/div[2]').inner_text().split()[0].strip()
            except:
                hotel_dict['reviews count'] = 'N/A'

            hotels_list.append(hotel_dict)

        df = pd.DataFrame(hotels_list)
        df.to_excel('hotels_list.xlsx', index=False)
        df.to_csv('hotels_list.csv', index=False)

        browser.close()


if __name__ == '__main__':
    main()
