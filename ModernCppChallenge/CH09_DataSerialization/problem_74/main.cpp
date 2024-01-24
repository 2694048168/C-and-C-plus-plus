/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Selecting data from XML using XPath
 * @version 0.1
 * @date 2024-01-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pugixml/pugixml.hpp"

#include <iostream>
#include <string>

/**
 * @brief Selecting data from XML using XPath
 * 
 * Consider an XML file with a list of movies as described for the previous problem.
 * Write a program that can select and print the following:
 * 1. The title of all the movies released after a given year
 * 2. The name of the last actor in the casting list for each movie in the file
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */

// ------------------------------
int main(int argc, char **argv)
{
    std::string text = R"(
<?xml version="1.0"?>
<movies>
	<movie id="11001" title="The Matrix" year="1999" length="196">
		<cast>
			<role star="Keanu Reeves" name="Neo" />
			<role star="Laurence Fishburne" name="Morpheus" />
			<role star="Carrie-Anne Moss" name="Trinity" />
			<role star="Hugo Weaving" name="	Agent Smith" />
		</cast>
		<directors>
			<director name="Lana Wachowski" />
			<director name="Lilly Wachowski" />
		</directors>
		<writers>
			<writer name="Lana Wachowski" />
			<writer name="Lilly Wachowski" />
		</writers>
	</movie>
	<movie id="9871" title="Forrest Gump" year="1994" length="202">
		<cast>
			<role star="Tom Hanks" name="Forrest Gump" />
			<role star="Sally Field" name="Mrs. Gump" />
			<role star="Robin Wright" name="Jenny Curran" />
			<role star="Mykelti Williamson" name="Bubba Blue" />
		</cast>
		<directors>
			<director name="Robert Zemeckis" />
		</directors>
		<writers>
			<writer name="Winston Groom" />
			<writer name="Eric Roth" />
		</writers>
	</movie>
</movies>
)";

    pugi::xml_document doc;
    if (doc.load_string(text.c_str()))
    {
        try
        {
            auto titles = doc.select_nodes("/movies/movie[@year>1995]");

            for (auto it : titles)
            {
                std::cout << it.node().attribute("title").as_string() << std::endl;
            }
        }
        catch (const pugi::xpath_exception &e)
        {
            std::cout << e.result().description() << std::endl;
        }

        try
        {
            auto titles = doc.select_nodes("/movies/movie/cast/role[last()]");

            for (auto it : titles)
            {
                std::cout << it.node().attribute("star").as_string() << std::endl;
            }
        }
        catch (const pugi::xpath_exception &e)
        {
            std::cout << e.result().description() << std::endl;
        }
    }

    return 0;
}
