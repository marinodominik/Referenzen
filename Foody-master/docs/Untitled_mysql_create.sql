CREATE TABLE `User` (
	`userID` INT NOT NULL AUTO_INCREMENT,
	`firstName` VARCHAR(255) NOT NULL,
	`lastName` VARCHAR(255) NOT NULL,
	`street` INT NOT NULL,
	`buildingNr` VARCHAR(255) NOT NULL,
	`city` VARCHAR(255) NOT NULL,
	`zip` INT NOT NULL,
	`telephone` INT NOT NULL,
	`email` VARCHAR(255) NOT NULL UNIQUE,
	`password` VARCHAR(255) NOT NULL,
	PRIMARY KEY (`userID`)
);

CREATE TABLE `Order` (
	`orderID` INT NOT NULL AUTO_INCREMENT,
	`paymentID` INT NOT NULL,
	`userID` FLOAT NOT NULL,
	`restaurantEmail` VARCHAR(255) NOT NULL,
	`amountToPay` FLOAT NOT NULL,
	`orderDate` DATE NOT NULL,
	`inProduction` BOOLEAN NOT NULL,
	`isReadyForDelivery` BOOLEAN NOT NULL,
	`orderIsOnTheWay` BOOLEAN NOT NULL,
	`orderDelivered` BOOLEAN NOT NULL,
	`ratingID` INT,
	`customerZip` INT NOT NULL,
	PRIMARY KEY (`orderID`)
);

CREATE TABLE `Payment` (
	`paymentID` INT NOT NULL AUTO_INCREMENT,
	`paymentMethod` VARCHAR(255) NOT NULL,
	PRIMARY KEY (`paymentID`)
);

CREATE TABLE `Rating` (
	`ratingID` INT NOT NULL,
  `orderID` INT NOT NULL,
	`userID` INT NOT NULL,
	`comment` VARCHAR(255) NOT NULL,
	`stars` INT(1-5) NOT NULL,
	PRIMARY KEY (`ratingID`)
);

CREATE TABLE `Dish` (
	`dishName` VARCHAR(255) NOT NULL,
	`dishDescription` VARCHAR(255) NOT NULL,
	`dishPrice` FLOAT NOT NULL,
	`userID` INT NOT NULL
);

CREATE TABLE `Membership` (
	`type` VARCHAR(255) NOT NULL,
	`userID` VARCHAR(255) NOT NULL
);

ALTER TABLE `Order` ADD CONSTRAINT `Order_fk0` FOREIGN KEY (`paymentID`) REFERENCES `Payment`(`paymentID`);

ALTER TABLE `Order` ADD CONSTRAINT `Order_fk1` FOREIGN KEY (`userID`) REFERENCES `User`(`userID`);

ALTER TABLE `Order` ADD CONSTRAINT `Order_fk2` FOREIGN KEY (`ratingID`) REFERENCES `Rating`(`userID`);

ALTER TABLE `Rating` ADD CONSTRAINT `Rating_fk0` FOREIGN KEY (`userID`) REFERENCES `User`(`userID`);

ALTER TABLE `Dish` ADD CONSTRAINT `Dish_fk0` FOREIGN KEY (`userID`) REFERENCES `User`(`userID`);

ALTER TABLE `Membership` ADD CONSTRAINT `Membership_fk0` FOREIGN KEY (`userID`) REFERENCES `User`(`userID`);

