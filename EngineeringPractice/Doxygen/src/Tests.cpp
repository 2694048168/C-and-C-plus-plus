/**
 * @file Tests.cpp
 * @brief Unit tests for the application.
 *
 * This file contains a series of unit tests designed to verify the functionality
 * and correctness of the application's business logic.
 */

#include "BusinessLogic.h"
#include "gtest/gtest.h"

/**
 * @brief Tests the addition functionality of the business logic.
 */
TEST(BusinessLogicTest, TestAddition)
{
    BusinessLogic bl;
    EXPECT_EQ(bl.add(1, 1), 2);
}

/**
 * @brief Tests error handling when network failures occur.
 */
TEST(BusinessLogicTest, TestNetworkFailure)
{
    BusinessLogic bl;
    EXPECT_THROW(bl.performNetworkOperation(), NetworkException);
}
