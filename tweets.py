import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def scrape():
	chromeOptions = webdriver.ChromeOptions()
	prefs = {'profile.default_content_setting_values': {'images': 2}}
	chromeOptions.add_experimental_option("prefs",prefs)

	browser = webdriver.Chrome(executable_path='chromedriver.exe', chrome_options=chromeOptions)

	# url = u'https://twitter.com/Starbucks?lang=en'
	url = u'https://twitter.com/dunkindonuts'

	browser.get(url)

	time.sleep(5)

	body = browser.find_element_by_tag_name('body')

	def check_action(str):
		if (not str) or (str.isspace()):
			return '0'
		return str
	while True:
		for _ in range(0, 128):
			body.send_keys(Keys.PAGE_DOWN)
			time.sleep(0.15)
		dates = browser.find_elements_by_class_name('_timestamp')
		for date in dates:
			if '9 Jan 2017' in date.text: # 1 Aug 2016 : sbux ; 9 Jan 2017 : dnkn
				print('Scraping to file...')
				actions = browser.find_elements_by_class_name('ProfileTweet-actionCountForPresentation')
				with open('tweets.txt', 'w') as log:
					for date in dates:
						log.write(date.text + '\n')
					log.write('\n')
					for i, action in enumerate(actions):
						if (i + 1)%5 == 0 or (i + 1)%5 == 3:
							continue
						log.write(check_action(action.text.strip()) + ' ')
				browser.quit()
				print('Finished Scraping to file')
				return

	browser.quit()
	print('Finished without event')
	return

try:
	scrape()
	browser.quit()
except Exception as e:
	raise e